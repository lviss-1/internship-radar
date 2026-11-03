import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from radar.config import Settings, load_settings
from radar.normalize import Posting
from radar.scoring import Budget, Resumes
from radar.state import UsageLedger

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
FEED_URL = "https://feed.test/listings.json"
WEBHOOK_URL = "https://discord.com/api/webhooks/123/test-token"

# Shortly after the newest record in tests/fixtures/simplify_listings.json was posted
# (2026-11-02 11:00 ET, outside quiet hours).
NOW = datetime.fromtimestamp(1793635200, tz=UTC)

SWE_RESUME = """Jordan Example
B.S. Computer Science, expected May 2028
- Built a REST API in Python for course scheduling
- Automated weekly reporting with SQL and Python"""
PM_RESUME = """Jordan Example
B.S. Computer Science, expected May 2028
- Ran user interviews for a campus app
- Wrote product specs for a scheduling feature"""
RESUMES = Resumes(swe=SWE_RESUME, pm=PM_RESUME)


def anthropic_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / "anthropic" / name).read_text(encoding="utf-8"))


def message_json(model: str, payload: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """A Messages API response in the documented shape, carrying `payload` as JSON text."""
    message = anthropic_fixture("score_message.json")
    message["model"] = model
    message["content"][-1]["text"] = json.dumps(payload)
    message.update(overrides)
    return message


class FakeClaude:
    """Routes the real Anthropic SDK through a mock transport and records each request body."""

    def __init__(self, respond: Callable[[dict[str, Any]], httpx2.Response]) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            body = json.loads(request.content)
            self.requests.append(body)
            self.headers.append(dict(request.headers))
            return respond(body)

        self.client = anthropic.Anthropic(
            api_key="test-key",
            max_retries=0,
            http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        )


def make_budget(settings: Settings, ledger: UsageLedger | None = None) -> Budget:
    return Budget(
        ledger or UsageLedger(),
        settings.budget,
        settings.pricing,
        settings.scoring.worst_case_input_tokens,
    )


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr("radar.net.time.sleep", slept.append)
    monkeypatch.setattr("radar.discord.time.sleep", slept.append)
    monkeypatch.setattr("radar.sources.descriptions.time.sleep", slept.append)
    return slept


@pytest.fixture
def settings() -> Settings:
    loaded = load_settings(ROOT / "config" / "settings.yaml")
    simplify = loaded.sources.simplify.model_copy(update={"url": FEED_URL})
    return loaded.model_copy(
        update={"sources": loaded.sources.model_copy(update={"simplify": simplify})}
    )


@pytest.fixture
def raw_listings() -> list[dict[str, Any]]:
    return json.loads((FIXTURES / "simplify_listings.json").read_text(encoding="utf-8"))


def make_posting(**overrides: Any) -> Posting:
    fields: dict[str, Any] = {
        "source": "simplify",
        "source_id": "abc",
        "company": "Example Co",
        "title": "Software Engineer Intern",
        "locations": ["NYC"],
        "url": "https://example.com/jobs/1",
        "posted_at": NOW,
        "terms": ["Summer 2027"],
        "category": "Software",
        "active": True,
    }
    fields.update(overrides)
    return Posting(**fields)
