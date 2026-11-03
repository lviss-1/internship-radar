import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)

SEEN_VERSION = 2
SEEN_FILENAME = "seen.json"
USAGE_FILENAME = "usage.json"
LAG_FILENAME = "lag.json"
LAG_SAMPLES = 200
MEASURED_PAIRS_KEPT = 2000
GIT_TIMEOUT_SECONDS = 60
BOT_NAME = "github-actions[bot]"
BOT_EMAIL = "41898282+github-actions[bot]@users.noreply.github.com"


class StateError(Exception):
    pass


class UsageLedger(BaseModel):
    """Month-to-date API spend and today's call count. Numbers only, safe for a public repo."""

    version: Literal[1] = 1
    month: str = ""
    month_usd: float = Field(default=0.0, ge=0)
    day: str = ""
    day_calls: int = Field(default=0, ge=0)
    budget_alert_day: str = ""
    heartbeat_day: str = ""

    def roll(self, today: date) -> None:
        """Reset counters when the month or day changed since the last run."""
        month = today.strftime("%Y-%m")
        if self.month != month:
            self.month, self.month_usd = month, 0.0
        if self.day != today.isoformat():
            self.day, self.day_calls = today.isoformat(), 0

    def record(self, usd: float) -> None:
        self.month_usd = round(self.month_usd + usd, 6)
        self.day_calls += 1


class LagStats(BaseModel):
    """Detection-lag samples in minutes. Numbers, board names and posting IDs only."""

    version: Literal[1] = 1
    first_polled: dict[str, str] = Field(default_factory=dict)
    radar_minutes: dict[str, list[float]] = Field(default_factory=dict)
    simplify_delay_minutes: list[float] = Field(default_factory=list)
    measured_pairs: list[str] = Field(default_factory=list)

    def record_radar(self, source: str, minutes: float) -> None:
        samples = self.radar_minutes.setdefault(source, [])
        samples.append(round(minutes, 1))
        del samples[:-LAG_SAMPLES]

    def record_simplify_delay(self, simplify_key: str, minutes: float) -> None:
        if simplify_key in self.measured_pairs:
            return
        self.simplify_delay_minutes.append(round(minutes, 1))
        del self.simplify_delay_minutes[:-LAG_SAMPLES]
        self.measured_pairs.append(simplify_key)
        del self.measured_pairs[:-MEASURED_PAIRS_KEPT]


@dataclass
class RadarState:
    posted: set[str] = field(default_factory=set)
    scored_below: set[str] = field(default_factory=set)
    usage: UsageLedger = field(default_factory=UsageLedger)
    lag: LagStats = field(default_factory=LagStats)

    def handled(self, keys: Iterable[str]) -> bool:
        return any(key in self.posted or key in self.scored_below for key in keys)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateError(f"state file {path.name} is unreadable or not JSON") from exc


def _string_set(payload: dict[str, object], key: str, name: str) -> set[str]:
    values = payload.get(key)
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise StateError(f"state file {name} must hold a list of string IDs under {key!r}")
    return set(values)


def load_seen(path: Path) -> tuple[set[str], set[str]]:
    """Return (posted, scored_below). Missing means a first run; corrupt errors, never resets."""
    if not path.exists():
        return set(), set()
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise StateError(f"state file {path.name} has an unsupported format")
    match payload.get("version"):
        case 1:
            return _string_set(payload, "ids", path.name), set()
        case 2:
            return (
                _string_set(payload, "posted", path.name),
                _string_set(payload, "scored_below", path.name),
            )
        case _:
            raise StateError(f"state file {path.name} has an unsupported format")


def _load_model[M: BaseModel](path: Path, model: type[M]) -> M:
    if not path.exists():
        return model()
    try:
        return model.model_validate(_read_json(path))
    except ValidationError as exc:
        raise StateError(f"state file {path.name} has an unsupported format") from exc


def load_usage(path: Path) -> UsageLedger:
    return _load_model(path, UsageLedger)


def _write_atomic(path: Path, text: str) -> None:
    """Write via a temp file so a crash mid-write can't leave a truncated file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def save_state(directory: Path, state: RadarState) -> None:
    seen = {
        "version": SEEN_VERSION,
        "posted": sorted(state.posted),
        "scored_below": sorted(state.scored_below),
    }
    _write_atomic(directory / SEEN_FILENAME, json.dumps(seen, indent=1) + "\n")
    _write_atomic(directory / USAGE_FILENAME, state.usage.model_dump_json(indent=1) + "\n")
    _write_atomic(directory / LAG_FILENAME, state.lag.model_dump_json(indent=1) + "\n")


class StateStore(Protocol):
    def load(self) -> RadarState: ...

    def persist(self, state: RadarState, message: str) -> None: ...


class FileStateStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def load(self) -> RadarState:
        posted, scored_below = load_seen(self.directory / SEEN_FILENAME)
        usage = load_usage(self.directory / USAGE_FILENAME)
        lag = _load_model(self.directory / LAG_FILENAME, LagStats)
        return RadarState(posted=posted, scored_below=scored_below, usage=usage, lag=lag)

    def persist(self, state: RadarState, message: str) -> None:
        save_state(self.directory, state)


class GitStateStore(FileStateStore):
    """Stores state in a checkout of the `state` branch and pushes after every write.

    Pushing on each persist is what makes reserve-before-post safe: the reservation is
    durable on GitHub before anything reaches Discord.
    """

    def persist(self, state: RadarState, message: str) -> None:
        save_state(self.directory, state)
        self._git("add", SEEN_FILENAME, USAGE_FILENAME, LAG_FILENAME)
        if self._git("diff", "--cached", "--quiet", check=False).returncode == 0:
            return
        self._git("-c", f"user.name={BOT_NAME}", "-c", f"user.email={BOT_EMAIL}",
                  "commit", "--quiet", "-m", message)  # fmt: skip
        self._git("push", "--quiet", "origin", "HEAD:state")

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        command = ["git", "-C", str(self.directory), *args]
        subcommand = next(arg for arg in args if not arg.startswith("-") and "=" not in arg)
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise StateError(f"git {subcommand} timed out") from exc
        if check and result.returncode != 0:
            raise StateError(f"git {subcommand} failed with exit code {result.returncode}")
        return result
