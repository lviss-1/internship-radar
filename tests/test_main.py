import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import httpx2
import pytest
import respx
import yaml

from radar.config import Settings
from radar.main import (
    ERRORS_ENV,
    HEALTHCHECK_ENV,
    MATCHES_ENV,
    PING_USER_ENV,
    describe_exception,
    heartbeat_is_due,
    main,
    ping_healthcheck,
)
from radar.net import make_client
from radar.scoring import RESUME_ENV
from radar.state import FileStateStore, RadarState, StateError, UsageLedger
from tests.conftest import (
    FEED_URL,
    NOW,
    PM_RESUME,
    ROOT,
    SWE_RESUME,
    FakeClaude,
    anthropic_fixture,
    message_json,
)


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


MATCHES_URL = "https://discord.com/api/webhooks/1/matches-token"
ERRORS_URL = "https://discord.com/api/webhooks/2/errors-token"
A16Z_LIST = "https://boards-api.greenhouse.io/v1/boards/a16z/jobs"


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz: Any = None) -> "FixedDatetime":
        return cls.fromtimestamp(NOW.timestamp(), tz=tz)


def fake_claude(body: dict[str, Any]) -> httpx2.Response:
    if body["model"] == "claude-sonnet-5-5":
        return httpx2.Response(200, json=anthropic_fixture("drafts_message.json"))
    score = {
        "resume": "swe",
        "score": 85,
        "reasons": ["fits"],
        "gaps": [],
        "relocation_mentioned": False,
        "eligibility_flags": [],
    }
    return httpx2.Response(200, json=message_json("claude-haiku-5-5", score))


@pytest.fixture
def e2e_settings(tmp_path: Path) -> Path:
    raw = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    raw["sources"]["simplify"]["url"] = FEED_URL
    watchlist = tmp_path / "watchlist.yaml"
    a16z = {
        "name": "Andreessen Horowitz",
        "status": "verified",
        "board": "greenhouse",
        "slug": "a16z",
    }
    watchlist.write_text(yaml.safe_dump({"companies": [a16z]}))
    raw["watchlist_path"] = str(watchlist)
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.fixture
def live_like(monkeypatch: pytest.MonkeyPatch, raw_listings: list[dict[str, Any]]) -> Any:
    monkeypatch.setattr("radar.main.datetime", FixedDatetime)
    listing = {
        "jobs": [
            {
                "id": 8046730003,
                "title": "Research Intern - Frontier",
                "location": {"name": "New York, NY"},
                "absolute_url": "https://a16z.com/about/jobs/?gh_jid=8046730003",
                "first_published": "2026-10-31T12:00:00+00:00",
            }
        ]
    }
    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_URL).mock(return_value=httpx.Response(200, json=raw_listings))
        mock.get(A16Z_LIST).mock(return_value=httpx.Response(200, json=listing))
        mock.get(f"{A16Z_LIST}/8046730003").mock(
            return_value=httpx.Response(200, json={"content": "&lt;p&gt;Research.&lt;/p&gt;"})
        )
        yield mock


def test_full_run_through_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, e2e_settings: Path, live_like: Any
) -> None:
    for name, value in {
        "DISCORD_WEBHOOK_MATCHES": MATCHES_URL,
        "DISCORD_WEBHOOK_ERRORS": ERRORS_URL,
        "DISCORD_PING_USER_ID": "42",
        "HEALTHCHECK_PING_URL": HEALTHCHECK,
        "RESUME_SWE_TEXT": SWE_RESUME,
        "RESUME_PM_TEXT": PM_RESUME,
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        "radar.main.make_anthropic_client", lambda _: FakeClaude(fake_claude).client
    )
    matches = live_like.post(MATCHES_URL).mock(return_value=httpx.Response(200, json={}))
    errors = live_like.post(ERRORS_URL).mock(return_value=httpx.Response(200, json={}))
    ping = live_like.post(HEALTHCHECK).mock(return_value=httpx.Response(200))
    live_like.route().mock(return_value=httpx.Response(404))

    code = main(["--settings", str(e2e_settings), "--state-dir", str(tmp_path / "state")])

    state = FileStateStore(tmp_path / "state").load()
    footers = [
        json.loads(call.request.content)["embeds"][0]["footer"]["text"]
        for call in matches.calls
        if not json.loads(call.request.content)["embeds"][0]["title"].startswith("Drafts: ")
    ]
    assert code == 0
    assert footers.count("greenhouse") == 1
    assert "greenhouse:8046730003" in state.posted
    assert ping.call_count == 1
    assert errors.call_count == 1
    assert "Heartbeat" in json.loads(errors.calls[0].request.content)["content"]


def test_dry_run_never_calls_claude_discord_or_healthcheck(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, e2e_settings: Path, live_like: Any
) -> None:
    monkeypatch.setenv("HEALTHCHECK_PING_URL", HEALTHCHECK)

    def no_claude(_: Any) -> None:
        raise AssertionError("dry run must not build a Claude client")

    monkeypatch.setattr("radar.main.make_anthropic_client", no_claude)
    others = live_like.route().mock(return_value=httpx.Response(404))

    code = main(["--dry-run", "--settings", str(e2e_settings), "--state-dir", str(tmp_path / "s")])

    called_hosts = {call.request.url.host for call in others.calls}
    assert code == 0
    assert not called_hosts & {"discord.com", "hc-ping.com"}
    assert not (tmp_path / "s").exists()


def test_workflow_passes_every_secret_the_code_reads() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/radar.yml").read_text(encoding="utf-8"))
    env = next(
        step["env"] for step in workflow["jobs"]["run"]["steps"] if step.get("id") == "radar"
    )
    required = {
        MATCHES_ENV,
        ERRORS_ENV,
        PING_USER_ENV,
        HEALTHCHECK_ENV,
        "ANTHROPIC_API_KEY",
        *RESUME_ENV.values(),
    }

    for name in sorted(required):
        assert env.get(name) == f"${{{{ secrets.{name} }}}}", name
