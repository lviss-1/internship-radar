from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from radar.alerts import Alert, AlertKind, plan_alerts
from radar.ats import Board
from radar.config import Settings
from radar.discord import DiscordError
from radar.main import (
    RunSummary,
    alert_run_failure,
    collect_alerts,
    send_alerts,
    track_board_health,
)
from radar.sources.boards import BoardResult, WatchlistBoard
from radar.state import FileStateStore, RadarState, UsageLedger

T0 = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)
COOLDOWN = timedelta(hours=6)
FEED_DOWN = Alert("simplify_failed", AlertKind.CONDITION, "Simplify feed failed.")
POST_FAILED = Alert("post_failed", AlertKind.EVENT, "2 postings failed to send.")
TARGET = Alert("budget_target:2026-11", AlertKind.ONCE, "Spend passed the target.")


def plan(alerts: list[Alert], sent: dict[str, str], at: datetime, month: str = "2026-11") -> Any:
    return plan_alerts(alerts, sent, at, COOLDOWN, month)


def test_condition_repeats_only_after_cooldown() -> None:
    sent: dict[str, str] = {}

    first = plan([FEED_DOWN], sent, T0)
    soon = plan([FEED_DOWN], sent, T0 + timedelta(hours=5))
    later = plan([FEED_DOWN], sent, T0 + timedelta(hours=6))

    assert first == "• Simplify feed failed."
    assert soon is None
    assert later == "• Simplify feed failed."


def test_condition_sends_one_recovery_line_when_it_clears() -> None:
    sent: dict[str, str] = {}
    plan([FEED_DOWN], sent, T0)

    recovered = plan([], sent, T0 + timedelta(minutes=30))
    quiet = plan([], sent, T0 + timedelta(hours=1))

    assert recovered == "• Recovered: Simplify feed is reachable again."
    assert quiet is None
    assert sent == {}


def test_board_down_recovery_names_the_board() -> None:
    down = Alert("board_down:greenhouse/x", AlertKind.CONDITION, "Board down.")
    sent: dict[str, str] = {}
    plan([down], sent, T0)

    assert plan([], sent, T0 + timedelta(hours=1)) == (
        "• Recovered: Board greenhouse/x is responding again."
    )


def test_events_are_throttled_without_recovery_lines() -> None:
    sent: dict[str, str] = {}

    first = plan([POST_FAILED], sent, T0)
    repeat = plan([POST_FAILED], sent, T0 + timedelta(hours=1))
    gone = plan([], sent, T0 + timedelta(hours=2))
    expired = plan([], sent, T0 + timedelta(hours=7))

    assert first is not None and repeat is None and gone is None and expired is None
    assert sent == {}


def test_once_alerts_never_repeat_within_their_period() -> None:
    sent: dict[str, str] = {}

    assert plan([TARGET], sent, T0) == "• Spend passed the target."
    assert plan([TARGET], sent, T0 + timedelta(days=3)) is None
    assert plan([], sent, T0 + timedelta(days=4)) is None
    assert "budget_target:2026-11" in sent


def test_once_alerts_are_forgotten_when_the_month_rolls_over() -> None:
    sent = {"budget_target:2026-10": T0.isoformat()}

    plan([], sent, T0, month="2026-11")

    assert sent == {}


def test_several_alerts_share_one_message() -> None:
    message = plan([FEED_DOWN, POST_FAILED], {}, T0)

    assert message == "• Simplify feed failed.\n• 2 postings failed to send."


class FailingHook:
    def send(self, payload: dict[str, Any]) -> None:
        raise DiscordError("DISCORD_WEBHOOK_ERRORS: HTTP 500")


class RecordingHook:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload["content"])


def test_alert_is_not_recorded_when_discord_rejects_it(settings: Settings) -> None:
    state = RadarState(usage=UsageLedger(month="2026-11"))

    send_alerts([FEED_DOWN], state, FailingHook(), T0, settings)  # type: ignore[arg-type]

    assert state.usage.alerts_sent == {}


def test_sent_alert_is_recorded(settings: Settings) -> None:
    state = RadarState(usage=UsageLedger(month="2026-11"))
    hook = RecordingHook()

    send_alerts([FEED_DOWN], state, hook, T0, settings)  # type: ignore[arg-type]

    assert hook.sent == ["**Internship Radar alert**\n• Simplify feed failed."]
    assert state.usage.alerts_sent == {"simplify_failed": T0.isoformat()}


BOARDS = [
    WatchlistBoard("A", Board.GREENHOUSE, "a"),
    WatchlistBoard("B", Board.ASHBY, "b"),
]


def test_board_health_counts_consecutive_failures_and_resets() -> None:
    failures: dict[str, int] = {"greenhouse/removed": 9}
    failing = BoardResult(polled=["ashby/b"], failures={"greenhouse/a": "HTTP 404"})
    healthy = BoardResult(polled=["greenhouse/a", "ashby/b"])

    downs = [track_board_health(failures, BOARDS, failing, down_after=3) for _ in range(3)]

    assert downs[:2] == [{}, {}]
    assert downs[2] == {"greenhouse/a": 3}
    assert "greenhouse/removed" not in failures
    assert track_board_health(failures, BOARDS, healthy, down_after=3) == {}
    assert failures == {}


def test_run_failure_alerts_once_per_cooldown(tmp_path: Path) -> None:
    store = FileStateStore(tmp_path)
    hook = RecordingHook()

    alert_run_failure(store, hook, "StateError: git push failed", T0, COOLDOWN)  # type: ignore[arg-type]
    alert_run_failure(store, hook, "StateError: git push failed", T0 + timedelta(hours=1), COOLDOWN)  # type: ignore[arg-type]
    alert_run_failure(store, hook, "StateError: git push failed", T0 + timedelta(hours=7), COOLDOWN)  # type: ignore[arg-type]

    assert len(hook.sent) == 2
    assert "Run failed: StateError: git push failed" in hook.sent[0]


def test_run_failure_with_unreadable_state_always_alerts(tmp_path: Path) -> None:
    (tmp_path / "seen.json").write_text("{broken")
    hook = RecordingHook()

    for hours in (0, 1):
        alert_run_failure(
            FileStateStore(tmp_path),
            hook,
            "boom",
            T0 + timedelta(hours=hours),
            COOLDOWN,  # type: ignore[arg-type]
        )

    assert len(hook.sent) == 2


@pytest.mark.parametrize("hours_later", [0.5, 3])
def test_recovery_follows_a_run_failure(hours_later: float) -> None:
    sent = {"run_failed": T0.isoformat()}

    message = plan([], sent, T0 + timedelta(hours=hours_later))

    assert message == "• Recovered: Runs are succeeding again."


@pytest.mark.parametrize(("age_days", "expected"), [(None, False), (30, False), (46, True)])
def test_stale_watchlist_reminder(
    settings: Settings, age_days: float | None, expected: bool
) -> None:
    summary = RunSummary(watchlist_age_days=age_days)
    state = RadarState(usage=UsageLedger(month="2026-11"))

    categories = [alert.category for alert in collect_alerts(summary, state, settings)]

    assert ("watchlist_stale:2026-11" in categories) is expected


def test_stale_watchlist_reminder_is_monthly() -> None:
    stale = Alert("watchlist_stale:2026-11", AlertKind.ONCE, "Re-run discovery.")
    sent: dict[str, str] = {}

    assert plan([stale], sent, T0) is not None
    assert plan([stale], sent, T0 + timedelta(days=10)) is None
    plan([], sent, T0 + timedelta(days=30), month="2026-12")
    assert sent == {}
