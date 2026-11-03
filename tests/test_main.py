import logging
from datetime import UTC, datetime

import httpx
import pytest
import respx

from radar.config import Settings
from radar.main import describe_exception, heartbeat_is_due, main, ping_healthcheck
from radar.net import make_client
from radar.state import RadarState, StateError, UsageLedger
from tests.conftest import ROOT


def test_known_errors_keep_their_message() -> None:
    assert describe_exception(StateError("git push failed")) == "StateError: git push failed"


def test_unknown_errors_hide_their_message() -> None:
    try:
        raise ValueError("Resume text or a job description could be in here")
    except ValueError as exc:
        described = describe_exception(exc)

    assert "Resume" not in described
    assert described.startswith("unexpected ValueError at test_main.py:")


def test_missing_webhook_fails_cleanly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("DISCORD_WEBHOOK_MATCHES", raising=False)
    monkeypatch.delenv("DISCORD_WEBHOOK_ERRORS", raising=False)

    with caplog.at_level(logging.ERROR):
        code = main(["--settings", str(ROOT / "config" / "settings.yaml")])

    assert code == 1
    assert "DISCORD_WEBHOOK_ERRORS is not set" in caplog.text


HEALTHCHECK = "https://hc-ping.com/test-uuid"


@pytest.mark.parametrize(("succeeded", "path"), [(True, ""), (False, "/fail")])
def test_healthcheck_reports_outcome(
    monkeypatch: pytest.MonkeyPatch, succeeded: bool, path: str
) -> None:
    monkeypatch.setenv("HEALTHCHECK_PING_URL", HEALTHCHECK)
    with respx.mock(assert_all_called=False) as mock, make_client() as client:
        route = mock.post(f"{HEALTHCHECK}{path}").mock(return_value=httpx.Response(200))
        ping_healthcheck(client, succeeded=succeeded)

    assert route.call_count == 1


@pytest.mark.parametrize("value", ["", "http://hc-ping.com/test-uuid"])
def test_healthcheck_skips_missing_or_insecure_url(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, value: str
) -> None:
    monkeypatch.setenv("HEALTHCHECK_PING_URL", value)
    with respx.mock() as mock, make_client() as client, caplog.at_level(logging.WARNING):
        ping_healthcheck(client, succeeded=True)

    assert mock.calls.call_count == 0
    assert "skipping the healthcheck ping" in caplog.text


def test_healthcheck_failure_never_raises_or_leaks_url(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("HEALTHCHECK_PING_URL", HEALTHCHECK)
    with respx.mock() as mock, make_client() as client, caplog.at_level(logging.WARNING):
        mock.post(HEALTHCHECK).mock(side_effect=httpx.ConnectError(f"cannot reach {HEALTHCHECK}"))
        ping_healthcheck(client, succeeded=True)

    assert "healthcheck ping failed" in caplog.text
    assert "test-uuid" not in caplog.text


@pytest.mark.parametrize(
    ("utc_hour", "sent_day", "due"),
    [
        (12, "", False),
        (13, "", True),
        (13, "2026-11-02", False),
        (4, "2026-11-01", False),
    ],
)
def test_heartbeat_is_due_once_after_the_configured_hour(
    settings: Settings, utc_hour: int, sent_day: str, due: bool
) -> None:
    now = datetime(2026, 11, 2, utc_hour, 30, tzinfo=UTC)
    state = RadarState(usage=UsageLedger(heartbeat_day=sent_day))

    assert heartbeat_is_due(state, now, settings) is due
