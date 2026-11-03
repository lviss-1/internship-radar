import json
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import httpx2
import pytest
import respx

from radar.config import Settings
from radar.discord import DiscordError
from radar.main import AiConfig, Hooks, RunSummary, run
from radar.net import make_client
from radar.state import (
    SEEN_FILENAME,
    USAGE_FILENAME,
    FileStateStore,
    GitStateStore,
    RadarState,
    StateError,
    UsageLedger,
    load_seen,
)
from tests.conftest import FEED_URL, NOW, RESUMES, FakeClaude, anthropic_fixture, message_json

# Scores the fake model gives each company in tests/fixtures/simplify_listings.json.
SCORES = {
    "Andreessen Horowitz": 90,
    "Microsoft": 70,
    "Molina Healthcare": 40,
    "Nokia": 85,
    "Elucid": 65,
    "KnowBe4": 82,
}
REFUSED = {"Entrust"}
RELOCATION = {"KnowBe4"}


def fake_model(body: dict[str, Any]) -> httpx2.Response:
    if body["model"] == "claude-sonnet-5-5":
        return httpx2.Response(200, json=anthropic_fixture("drafts_message.json"))
    content = body["messages"][0]["content"]
    company = re.search(r"<company>(.*?)</company>", content).group(1)  # type: ignore[union-attr]
    if company in REFUSED:
        return httpx2.Response(200, json=anthropic_fixture("refusal_message.json"))
    payload = {
        "resume": "swe",
        "score": SCORES[company],
        "reasons": ["fits"],
        "gaps": [],
        "relocation_mentioned": company in RELOCATION,
        "eligibility_flags": [],
    }
    return httpx2.Response(200, json=message_json("claude-haiku-5-5", payload))


class RecordingHook:
    def __init__(self, fail_titles: set[str] | None = None, events: list[str] | None = None):
        self.sent: list[dict[str, Any]] = []
        self.fail_titles = fail_titles or set()
        self.events = events if events is not None else []

    def send(self, payload: dict[str, Any]) -> None:
        title = payload["embeds"][0]["title"] if payload.get("embeds") else payload["content"]
        self.events.append(f"send {title}")
        if any(fragment in title for fragment in self.fail_titles):
            raise DiscordError("DISCORD_WEBHOOK_MATCHES: HTTP 500")
        self.sent.append(payload)

    def cards(self) -> list[dict[str, Any]]:
        return [p for p in self.sent if not p["embeds"][0]["title"].startswith("Drafts: ")]


class RecordingStore(FileStateStore):
    def __init__(self, directory: Path, events: list[str]):
        super().__init__(directory)
        self.events = events

    def persist(self, state: RadarState, message: str) -> None:
        self.events.append(f"persist posted={len(state.posted)} below={len(state.scored_below)}")
        super().persist(state, message)


@pytest.fixture
def feed(raw_listings: list[dict[str, Any]]) -> Any:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_URL).mock(return_value=httpx.Response(200, json=raw_listings))
        mock.route().mock(return_value=httpx.Response(404))
        yield mock


def run_once(
    settings: Settings,
    store: FileStateStore,
    matches: RecordingHook,
    errors: RecordingHook | None = None,
    claude: FakeClaude | None = None,
) -> RunSummary:
    claude = claude or FakeClaude(fake_model)
    hooks = Hooks(matches=matches, errors=errors or RecordingHook(), ping_user_id="42")  # type: ignore[arg-type]
    with make_client() as client:
        return run(settings, client, store, hooks, AiConfig(claude.client, RESUMES), NOW)


def card_titles(hook: RecordingHook) -> list[str]:
    return [payload["embeds"][0]["title"] for payload in hook.cards()]


def test_missing_state_is_a_first_run(tmp_path: Path) -> None:
    state = FileStateStore(tmp_path).load()

    assert (state.posted, state.scored_below, state.usage) == (set(), set(), UsageLedger())


def test_version_1_state_migrates(tmp_path: Path) -> None:
    (tmp_path / SEEN_FILENAME).write_text(json.dumps({"version": 1, "ids": ["simplify:a"]}))

    assert load_seen(tmp_path / SEEN_FILENAME) == ({"simplify:a"}, set())


def test_round_trip_holds_ids_and_numbers_only(tmp_path: Path) -> None:
    store = FileStateStore(tmp_path)
    usage = UsageLedger(month="2026-11", month_usd=1.25, day="2026-11-02", day_calls=3)
    store.persist(RadarState({"simplify:b", "simplify:a"}, {"simplify:c"}, usage), "save")

    seen = json.loads((tmp_path / SEEN_FILENAME).read_text())
    assert seen == {
        "version": 2,
        "posted": ["simplify:a", "simplify:b"],
        "scored_below": ["simplify:c"],
    }
    assert store.load().usage == usage
    assert sorted(p.name for p in tmp_path.iterdir()) == [SEEN_FILENAME, USAGE_FILENAME]


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        json.dumps({"version": 3, "posted": []}),
        json.dumps({"version": 2, "posted": [1], "scored_below": []}),
        json.dumps({"version": 2, "posted": []}),
        json.dumps(["simplify:a"]),
    ],
)
def test_corrupt_state_raises_instead_of_resetting(tmp_path: Path, content: str) -> None:
    (tmp_path / SEEN_FILENAME).write_text(content)

    with pytest.raises(StateError):
        FileStateStore(tmp_path).load()


def test_corrupt_usage_raises(tmp_path: Path) -> None:
    (tmp_path / USAGE_FILENAME).write_text(json.dumps({"version": 1, "month_usd": -5}))

    with pytest.raises(StateError):
        FileStateStore(tmp_path).load()


def test_ledger_resets_on_new_day_and_month() -> None:
    ledger = UsageLedger(month="2026-10", month_usd=9.5, day="2026-10-31", day_calls=150)

    ledger.roll(date(2026, 10, 31))
    assert (ledger.month_usd, ledger.day_calls) == (9.5, 150)

    ledger.roll(date(2026, 11, 1))
    assert (ledger.month, ledger.month_usd, ledger.day, ledger.day_calls) == (
        "2026-11", 0.0, "2026-11-01", 0,
    )  # fmt: skip


def test_scored_pipeline_posts_and_records_correctly(
    settings: Settings, tmp_path: Path, feed: Any
) -> None:
    store = FileStateStore(tmp_path)
    matches = RecordingHook()
    claude = FakeClaude(fake_model)

    summary = run_once(settings, store, matches, claude=claude)

    state = store.load()
    cards = {p["embeds"][0]["title"].split(" — ")[0]: p for p in matches.cards()}
    drafts = [p for p in matches.sent if p["embeds"][0]["title"].startswith("Drafts: ")]
    assert set(cards) == {
        "Andreessen Horowitz",
        "Microsoft",
        "Nokia",
        "Elucid",
        "KnowBe4",
        "Entrust",
    }
    assert cards["Andreessen Horowitz"].get("content") == "<@42>"
    assert cards["KnowBe4"].get("content") == "<@42>"
    assert "content" not in cards["Nokia"] and cards["Nokia"]["flags"] != 0
    assert cards["Entrust"]["embeds"][0]["fields"][0]["value"] == "Unscored (refused)"
    assert len(drafts) == 2
    assert len(state.scored_below) == 1 and len(state.posted) == 6
    assert (summary.scored, summary.drafted, summary.dropped_edits) == (6, 2, 2)
    assert summary.unscored == {"refused": 1}
    assert state.usage.day_calls == 9
    assert state.usage.month_usd > 0
    assert len([r for r in claude.requests if r["model"] == "claude-sonnet-5-5"]) == 2


def test_three_consecutive_runs_never_rescore_or_repost(
    settings: Settings, tmp_path: Path, feed: Any
) -> None:
    store = FileStateStore(tmp_path)
    matches = RecordingHook()
    claude = FakeClaude(fake_model)

    summaries = [run_once(settings, store, matches, claude=claude) for _ in range(3)]

    titles = card_titles(matches)
    assert [s.posted for s in summaries] == [6, 0, 0]
    assert summaries[1].already_handled == summaries[2].already_handled == 7
    assert len(titles) == len(set(titles)) == 6
    assert len(claude.requests) == 9


def test_cap_defers_the_rest_without_duplicates(
    settings: Settings, tmp_path: Path, feed: Any
) -> None:
    store = FileStateStore(tmp_path)
    matches = RecordingHook()
    capped = settings.model_copy(
        update={"delivery": settings.delivery.model_copy(update={"max_posts_per_run": 3})}
    )

    summaries = [run_once(capped, store, matches) for _ in range(4)]

    titles = card_titles(matches)
    assert sum(s.posted + s.below_threshold for s in summaries) == 7
    assert summaries[-1].posted == summaries[-1].below_threshold == 0
    assert len(titles) == len(set(titles)) == 6


def test_ids_are_reserved_before_anything_is_posted(
    settings: Settings, tmp_path: Path, feed: Any
) -> None:
    events: list[str] = []

    run_once(settings, RecordingStore(tmp_path, events), RecordingHook(events=events))

    assert events[0] == "persist posted=6 below=1"
    assert all(event.startswith("send") for event in events[1:])


def test_failed_posts_are_released_and_retried_without_rescoring(
    settings: Settings, tmp_path: Path, feed: Any
) -> None:
    events: list[str] = []
    store = RecordingStore(tmp_path, events)

    first = run_once(settings, store, RecordingHook(fail_titles={"Elucid"}, events=events))

    assert (first.posted, first.failed) == (5, 1)
    assert events[-1] == "persist posted=5 below=1"

    claude = FakeClaude(fake_model)
    retry = RecordingHook()
    second = run_once(settings, store, retry, claude=claude)

    assert second.posted == 1
    assert "Elucid" in card_titles(retry)[0]
    assert len(claude.requests) == 1


def test_consecutive_discord_failures_stop_and_release_everything(
    settings: Settings, tmp_path: Path, feed: Any
) -> None:
    store = FileStateStore(tmp_path)
    matches = RecordingHook(fail_titles={" — "})

    summary = run_once(settings, store, matches)

    sends = [event for event in matches.events if event.startswith("send")]
    assert len(sends) == 3
    assert (summary.posted, summary.failed) == (0, 6)
    assert store.load().posted == set()
    assert len(store.load().scored_below) == 1


def test_budget_exhaustion_posts_unscored_and_alerts_once_per_day(
    settings: Settings, tmp_path: Path, feed: Any
) -> None:
    store = FileStateStore(tmp_path)
    exhausted = UsageLedger(
        month="2026-11", month_usd=settings.budget.monthly_usd_hard_cap, day="2026-11-02"
    )
    store.persist(RadarState(usage=exhausted), "seed")
    errors = RecordingHook()
    matches = RecordingHook()
    claude = FakeClaude(fake_model)

    summary = run_once(settings, store, matches, errors=errors, claude=claude)

    assert claude.requests == []
    assert summary.posted == 7 and summary.unscored == {"budget": 7}
    assert all(card["flags"] != 0 for card in matches.cards())
    assert len(errors.sent) == 1 and "monthly budget cap" in errors.sent[0]["content"]

    store.persist(RadarState(usage=store.load().usage), "forget posts, keep alert day")
    run_once(settings, store, RecordingHook(), errors=errors, claude=claude)
    assert len(errors.sent) == 1


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


def test_git_store_pushes_both_files_and_skips_empty_commits(
    state_checkout: tuple[Path, Path],
) -> None:
    remote, checkout = state_checkout
    store = GitStateStore(checkout)
    usage = UsageLedger(month="2026-11", month_usd=0.5)

    store.persist(RadarState({"simplify:a"}, set(), usage), "reserve 1")
    store.persist(RadarState({"simplify:a"}, set(), usage), "no-op")
    store.persist(RadarState({"simplify:a", "simplify:b"}, {"simplify:c"}, usage), "reserve 1")

    seen = json.loads(git("show", f"state:{SEEN_FILENAME}", cwd=remote))
    pushed_usage = json.loads(git("show", f"state:{USAGE_FILENAME}", cwd=remote))
    assert seen["posted"] == ["simplify:a", "simplify:b"] and seen["scored_below"] == ["simplify:c"]
    assert pushed_usage["month_usd"] == 0.5
    assert git("rev-list", "--count", "state", cwd=remote) == "2"


def test_git_store_push_failure_is_a_state_error(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    git("init", "--quiet", "-b", "state", str(checkout), cwd=tmp_path)
    git("remote", "add", "origin", str(tmp_path / "missing.git"), cwd=checkout)

    with pytest.raises(StateError, match="git push failed"):
        GitStateStore(checkout).persist(RadarState({"simplify:a"}), "reserve 1")
