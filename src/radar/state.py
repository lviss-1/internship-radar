import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

STATE_VERSION = 1
STATE_FILENAME = "seen.json"
GIT_TIMEOUT_SECONDS = 60
BOT_NAME = "github-actions[bot]"
BOT_EMAIL = "41898282+github-actions[bot]@users.noreply.github.com"


class StateError(Exception):
    pass


def load_ids(path: Path) -> set[str]:
    """Load posted IDs. A missing file means a first run; a corrupt file errors, never resets."""
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateError(f"state file {path.name} is unreadable or not JSON") from exc

    if not isinstance(payload, dict) or payload.get("version") != STATE_VERSION:
        raise StateError(f"state file {path.name} has an unsupported format")
    ids = payload.get("ids")
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise StateError(f"state file {path.name} must hold a list of string IDs")
    return set(ids)


def save_ids(path: Path, ids: set[str]) -> None:
    """Write atomically so a crash mid-write can't leave a truncated file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"version": STATE_VERSION, "ids": sorted(ids)}, indent=1) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


class StateStore(Protocol):
    def load(self) -> set[str]: ...

    def persist(self, ids: set[str], message: str) -> None: ...


class FileStateStore:
    def __init__(self, directory: Path) -> None:
        self.path = directory / STATE_FILENAME

    def load(self) -> set[str]:
        return load_ids(self.path)

    def persist(self, ids: set[str], message: str) -> None:
        save_ids(self.path, ids)


class GitStateStore(FileStateStore):
    """Stores state in a checkout of the `state` branch and pushes after every write.

    Pushing on each persist is what makes reserve-before-post safe: the reservation is
    durable on GitHub before anything reaches Discord.
    """

    def __init__(self, directory: Path) -> None:
        super().__init__(directory)
        self.directory = directory

    def persist(self, ids: set[str], message: str) -> None:
        save_ids(self.path, ids)
        self._git("add", STATE_FILENAME)
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
