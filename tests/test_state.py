import json
import re
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import httpx2
import pytest
import respx

from radar.ats import Board
from radar.config import Settings
from radar.discord import DiscordError
from radar.feedback import FeedbackConfig
from radar.main import AiConfig, Hooks, RunSummary, collect_alerts, record_radar_lag, run
from radar.net import make_client
from radar.sources.boards import WatchlistBoard
from radar.state import (
    LAG_FILENAME,
    SEEN_FILENAME,
    USAGE_FILENAME,
    FileStateStore,
    GitStateStore,
    LagStats,
    RadarState,
    StateError,
    UsageLedger,
    load_seen,
)
from tests.conftest import (
    FEED_URL,
    NOW,
    RESUMES,
    FakeClaude,
    anthropic_fixture,
    make_posting,
    message_json,
)

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
        self.events.append(
            f"persist posted={own_keys(state.posted)} below={own_keys(state.scored_below)}"
        )
        super().persist(state, message)


def own_keys(keys: set[str]) -> int:
    """Count postings by their source key; aliases like greenhouse:123 ride along."""
    return sum(key.startswith("simplify:") for key in keys)


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
    boards: list[WatchlistBoard] | None = None,
    feedback: FeedbackConfig | None = None,
) -> RunSummary:
    claude = claude or FakeClaude(fake_model)
    hooks = Hooks(
        matches=matches,  # type: ignore[arg-type]
        errors=errors or RecordingHook(),  # type: ignore[arg-type]
        ping_user_id="42",
        feedback=feedback,
    )
    with make_client() as client:
        return run(
            settings, client, store, hooks, AiConfig(claude.client, RESUMES), NOW, boards=boards
        )


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
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        LAG_FILENAME,
        SEEN_FILENAME,
        USAGE_FILENAME,
    ]


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
    assert own_keys(state.scored_below) == 1 and own_keys(state.posted) == 6
    assert {key for key in state.posted if key.startswith("greenhouse:")}
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

    sends = [i for i, event in enumerate(events) if event.startswith("send")]
    assert events[0] == "persist posted=6 below=1"
    assert sends and all(events[i].startswith("send") for i in range(sends[0], sends[-1] + 1))


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
    assert own_keys(store.load().scored_below) == 1


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
    assert len(errors.sent) == 1
    assert "monthly budget cap" in errors.sent[0]["content"]
    assert "past the $10.00 target" in errors.sent[0]["content"]

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


def test_heartbeat_is_due_on_first_run_of_the_day_only(
    settings: Settings, tmp_path: Path, feed: Any
) -> None:
    store = FileStateStore(tmp_path)

    first = run_once(settings, store, RecordingHook())
    second = run_once(settings, store, RecordingHook())

    assert (first.heartbeat_due, second.heartbeat_due) == (True, False)
    assert store.load().usage.heartbeat_day == "2026-11-02"


A16Z = WatchlistBoard("Andreessen Horowitz", Board.GREENHOUSE, "a16z")
A16Z_JOB = 8046730003
A16Z_SIMPLIFY_POSTED = datetime.fromtimestamp(1793401464, tz=UTC)
GH = "https://boards-api.greenhouse.io/v1/boards"


@contextmanager
def sources(raw_listings: list[dict[str, Any]], **overrides: httpx.Response) -> Iterator[None]:
    """Mock the feed and the a16z board; `overrides` replaces either by name."""
    listing = {
        "jobs": [
            {
                "id": A16Z_JOB,
                "title": "Research Intern - Frontier",
                "location": {"name": "New York, NY"},
                "absolute_url": f"https://a16z.com/about/jobs/?gh_jid={A16Z_JOB}",
                "first_published": (A16Z_SIMPLIFY_POSTED - timedelta(hours=3)).isoformat(),
            }
        ]
    }
    detail = {"content": "&lt;p&gt;Research the frontier.&lt;/p&gt;", "offices": []}
    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_URL).mock(
            return_value=overrides.get("feed", httpx.Response(200, json=raw_listings))
        )
        mock.get(f"{GH}/a16z/jobs").mock(
            return_value=overrides.get("board", httpx.Response(200, json=listing))
        )
        mock.get(f"{GH}/a16z/jobs/{A16Z_JOB}").mock(return_value=httpx.Response(200, json=detail))
        mock.route().mock(return_value=httpx.Response(404))
        yield


def test_board_copy_wins_and_simplify_copy_never_posts(
    settings: Settings, tmp_path: Path, raw_listings: list[dict[str, Any]]
) -> None:
    store = FileStateStore(tmp_path)
    first, second = RecordingHook(), RecordingHook()

    with sources(raw_listings):
        run_once(settings, store, first, boards=[A16Z])
        run_once(settings, store, second, boards=[A16Z])

    a16z_cards = [c for c in first.cards() if c["embeds"][0]["title"].startswith("Andreessen")]
    state = store.load()
    assert len(a16z_cards) == 1
    assert a16z_cards[0]["embeds"][0]["footer"]["text"] == "greenhouse"
    assert f"greenhouse:{A16Z_JOB}" in state.posted
    assert len(first.cards()) == 6 and second.cards() == []
    assert state.lag.simplify_delay_minutes == [180.0]
    assert state.lag.first_polled.keys() == {"simplify", "greenhouse/a16z"}


def test_failing_board_is_reported_without_stopping_simplify(
    settings: Settings, tmp_path: Path, raw_listings: list[dict[str, Any]]
) -> None:
    with sources(raw_listings, board=httpx.Response(404)):
        summary = run_once(settings, FileStateStore(tmp_path), RecordingHook(), boards=[A16Z])

    assert summary.posted == 6
    assert summary.board_failures == {"greenhouse/a16z": "HTTP 404"}
    assert summary.source_failed
    messages = [alert.message for alert in collect_alerts(summary, RadarState(), settings)]
    assert "1 of 1 boards failed this run (greenhouse/a16z: HTTP 404)." in messages


def test_failing_simplify_feed_does_not_stop_boards(
    settings: Settings, tmp_path: Path, raw_listings: list[dict[str, Any]]
) -> None:
    matches = RecordingHook()

    with sources(raw_listings, feed=httpx.Response(500)):
        summary = run_once(settings, FileStateStore(tmp_path), matches, boards=[A16Z])

    assert summary.simplify_failure is not None
    assert summary.source_failed
    assert [c["embeds"][0]["footer"]["text"] for c in matches.cards()] == ["greenhouse"]
    categories = [alert.category for alert in collect_alerts(summary, RadarState(), settings)]
    assert "simplify_failed" in categories


def test_radar_lag_ignores_backlog_from_before_the_first_poll() -> None:
    lag = LagStats(first_polled={"greenhouse/a16z": "2026-11-02T12:00:00+00:00"})
    backlog = make_board_posting(posted_at=datetime(2026, 11, 2, 11, 0, tzinfo=UTC))
    fresh = make_board_posting(posted_at=datetime(2026, 11, 2, 15, 30, tzinfo=UTC))

    record_radar_lag(backlog, lag, NOW)
    record_radar_lag(fresh, lag, NOW)

    assert lag.radar_minutes == {"greenhouse": [30.0]}


def test_lag_samples_are_capped_and_pairs_counted_once() -> None:
    lag = LagStats()
    for minute in range(250):
        lag.record_radar("greenhouse", float(minute))
    lag.record_simplify_delay("simplify:a", 60.0)
    lag.record_simplify_delay("simplify:a", 90.0)

    assert len(lag.radar_minutes["greenhouse"]) == 200
    assert lag.radar_minutes["greenhouse"][0] == 50.0
    assert lag.simplify_delay_minutes == [60.0]


def make_board_posting(posted_at: datetime) -> Any:
    return make_posting(
        source="greenhouse", source_id="1", board="greenhouse/a16z", posted_at=posted_at
    )


def test_board_down_alert_then_recovery_across_runs(
    settings: Settings, tmp_path: Path, raw_listings: list[dict[str, Any]]
) -> None:
    store = FileStateStore(tmp_path)
    errors = RecordingHook()

    with sources(raw_listings, board=httpx.Response(404)):
        for _ in range(3):
            run_once(settings, store, RecordingHook(), errors=errors, boards=[A16Z])
    with sources(raw_listings):
        run_once(settings, store, RecordingHook(), errors=errors, boards=[A16Z])

    messages = [payload["content"] for payload in errors.sent]
    assert len(messages) == 3
    assert "1 of 1 boards failed this run" in messages[0]
    assert "Board greenhouse/a16z has failed 3 runs in a row (HTTP 404)" in messages[1]
    assert "Recovered: Board greenhouse/a16z is responding again." in messages[2]
    assert "Recovered: Board polling is healthy again." in messages[2]
    assert store.load().lag.board_failures == {}


def test_postings_handled_before_aliases_existed_are_not_reposted_from_boards(
    settings: Settings, tmp_path: Path, raw_listings: list[dict[str, Any]]
) -> None:
    a16z_simplify = next(r for r in raw_listings if r["company_name"] == "Andreessen Horowitz")
    store = FileStateStore(tmp_path)
    store.persist(RadarState(posted={f"simplify:{a16z_simplify['id']}"}), "legacy, no aliases")
    matches = RecordingHook()

    with sources(raw_listings):
        run_once(settings, store, matches, boards=[A16Z])

    titles = [card["embeds"][0]["title"] for card in matches.cards()]
    assert not any(title.startswith("Andreessen Horowitz") for title in titles)
    assert f"greenhouse:{A16Z_JOB}" in store.load().posted


def test_backfill_covers_both_buckets_and_reports_count() -> None:
    state = RadarState(posted={"simplify:a"}, scored_below={"simplify:b"})

    added = state.backfill_aliases(
        [
            ("simplify:a", frozenset({"simplify:a", "greenhouse:1"})),
            ("simplify:b", frozenset({"simplify:b", "ashby:x"})),
            ("simplify:c", frozenset({"simplify:c", "lever:y"})),
        ]
    )

    assert added == 2
    assert state.posted == {"simplify:a", "greenhouse:1"}
    assert state.scored_below == {"simplify:b", "ashby:x"}


def test_feedback_failure_alerts_once_and_scoring_continues(
    settings: Settings, tmp_path: Path, raw_listings: list[dict[str, Any]]
) -> None:
    config = FeedbackConfig(bot_token="t", webhook_url="https://discord.com/api/webhooks/9/x")
    errors = RecordingHook()

    with sources(raw_listings):
        summary = run_once(settings, FileStateStore(tmp_path), RecordingHook(), errors=errors,
                           feedback=config)  # fmt: skip

    assert summary.scored == 6 and summary.posted == 6
    assert summary.feedback_error == "matches webhook: HTTP 404"
    assert (
        "Reaction feedback is off this run (matches webhook: HTTP 404)" in errors.sent[0]["content"]
    )
