import json
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from radar.config import Settings
from radar.discord import DiscordError
from radar.main import run
from radar.net import make_client
from radar.state import (
    STATE_FILENAME,
    FileStateStore,
    GitStateStore,
    StateError,
    load_ids,
    save_ids,
)
from tests.conftest import FEED_URL, NOW


class RecordingHook:
    """Stands in for DiscordWebhook and records what would have been posted."""

    def __init__(self, fail_titles: set[str] | None = None, events: list[str] | None = None):
        self.sent: list[dict[str, Any]] = []
        self.fail_titles = fail_titles or set()
        self.events = events if events is not None else []

    def send(self, payload: dict[str, Any]) -> None:
        title = payload["embeds"][0]["title"]
        self.events.append(f"send {title}")
        if any(fragment in title for fragment in self.fail_titles):
            raise DiscordError("DISCORD_WEBHOOK_MATCHES: HTTP 500")
        self.sent.append(payload)


class RecordingStore(FileStateStore):
    def __init__(self, directory: Path, events: list[str]):
        super().__init__(directory)
        self.events = events
        self.persist_count = 0

    def persist(self, ids: set[str], message: str) -> None:
        self.events.append(f"persist {len(ids)}")
        self.persist_count += 1
        super().persist(ids, message)


def with_cap(settings: Settings, cap: int) -> Settings:
    delivery = settings.delivery.model_copy(update={"max_posts_per_run": cap})
    return settings.model_copy(update={"delivery": delivery})


def run_once(settings: Settings, store: FileStateStore, hook: RecordingHook) -> Any:
    with make_client() as client:
        return run(settings, client, store, hook, NOW)  # type: ignore[arg-type]


@pytest.fixture
def feed(raw_listings: list[dict[str, Any]]) -> Any:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_URL).mock(return_value=httpx.Response(200, json=raw_listings))
        yield mock


def test_missing_state_file_is_a_first_run(tmp_path: Path) -> None:
    assert load_ids(tmp_path / STATE_FILENAME) == set()


def test_round_trip_is_sorted_ids_only(tmp_path: Path) -> None:
    path = tmp_path / STATE_FILENAME
    save_ids(path, {"simplify:b", "simplify:a"})

    assert load_ids(path) == {"simplify:a", "simplify:b"}
    assert json.loads(path.read_text()) == {"version": 1, "ids": ["simplify:a", "simplify:b"]}
    assert [p.name for p in tmp_path.iterdir()] == [STATE_FILENAME]


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        json.dumps({"version": 2, "ids": []}),
        json.dumps({"version": 1, "ids": [1, 2]}),
        json.dumps({"version": 1}),
        json.dumps(["simplify:a"]),
    ],
)
def test_corrupt_state_raises_instead_of_resetting(tmp_path: Path, content: str) -> None:
    path = tmp_path / STATE_FILENAME
    path.write_text(content)

    with pytest.raises(StateError):
        load_ids(path)


def test_three_consecutive_runs_never_duplicate(
    settings: Settings, tmp_path: Path, feed: Any
) -> None:
    store = FileStateStore(tmp_path)
    hook = RecordingHook()

    first = run_once(settings, store, hook)
    second = run_once(settings, store, hook)
    third = run_once(settings, store, hook)

    titles = [payload["embeds"][0]["title"] for payload in hook.sent]
    assert first.posted == 7
    assert (second.posted, third.posted) == (0, 0)
    assert second.already_posted == third.already_posted == 7
    assert len(titles) == len(set(titles)) == 7


def test_cap_defers_the_rest_without_duplicates(
    settings: Settings, tmp_path: Path, feed: Any
) -> None:
    store = FileStateStore(tmp_path)
    hook = RecordingHook()
    capped = with_cap(settings, 3)

    posted = [run_once(capped, store, hook).posted for _ in range(4)]

    titles = [payload["embeds"][0]["title"] for payload in hook.sent]
    assert posted == [3, 3, 1, 0]
    assert len(titles) == len(set(titles)) == 7


def test_ids_are_reserved_before_anything_is_posted(
    settings: Settings, tmp_path: Path, feed: Any
) -> None:
    events: list[str] = []
    store = RecordingStore(tmp_path, events)

    run_once(settings, store, RecordingHook(events=events))

    assert events[0] == "persist 7"
    assert all(event.startswith("send") for event in events[1:])
    assert store.persist_count == 1


def test_failed_posts_are_released_and_retried_next_run(
    settings: Settings, tmp_path: Path, feed: Any
) -> None:
    events: list[str] = []
    store = RecordingStore(tmp_path, events)

    first = run_once(settings, store, RecordingHook(fail_titles={"Elucid"}, events=events))

    assert (first.posted, first.failed) == (6, 1)
    assert events[0] == "persist 7"
    assert events[-1] == "persist 6"

    retry_hook = RecordingHook()
    second = run_once(settings, store, retry_hook)

    assert second.posted == 1
    assert "Elucid" in retry_hook.sent[0]["embeds"][0]["title"]


def test_stops_after_consecutive_failures_and_releases_everything(
    settings: Settings, tmp_path: Path, feed: Any
) -> None:
    store = FileStateStore(tmp_path)
    hook = RecordingHook(fail_titles={" — "})

    summary = run_once(settings, store, hook)

    sends = [event for event in hook.events if event.startswith("send")]
    assert len(sends) == 3
    assert (summary.posted, summary.failed) == (0, 7)
    assert store.load() == set()


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def state_checkout(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    checkout = tmp_path / "checkout"
    git("init", "--quiet", "--bare", str(remote), cwd=tmp_path)
    git("init", "--quiet", "-b", "state", str(checkout), cwd=tmp_path)
    git("remote", "add", "origin", str(remote), cwd=checkout)
    return remote, checkout


def test_git_store_pushes_to_state_branch(state_checkout: tuple[Path, Path]) -> None:
    remote, checkout = state_checkout
    store = GitStateStore(checkout)

    store.persist({"simplify:a"}, "reserve 1 postings")
    store.persist({"simplify:a"}, "no-op")
    store.persist({"simplify:a", "simplify:b"}, "reserve 1 postings")

    pushed = json.loads(git("show", f"state:{STATE_FILENAME}", cwd=remote))
    assert pushed["ids"] == ["simplify:a", "simplify:b"]
    assert git("rev-list", "--count", "state", cwd=remote) == "2"


def test_git_store_push_failure_is_a_state_error(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    git("init", "--quiet", "-b", "state", str(checkout), cwd=tmp_path)
    git("remote", "add", "origin", str(tmp_path / "missing.git"), cwd=checkout)

    with pytest.raises(StateError, match="git push failed"):
        GitStateStore(checkout).persist({"simplify:a"}, "reserve 1 postings")
