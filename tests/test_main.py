import logging

import pytest

from radar.main import describe_exception, main
from radar.state import StateError
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
