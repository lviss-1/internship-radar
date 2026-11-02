import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from radar.config import Settings, load_settings
from radar.normalize import Posting

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
FEED_URL = "https://feed.test/listings.json"
WEBHOOK_URL = "https://discord.com/api/webhooks/123/test-token"

# Shortly after the newest record in tests/fixtures/simplify_listings.json was posted.
NOW = datetime.fromtimestamp(1793635200, tz=UTC)


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr("radar.net.time.sleep", slept.append)
    monkeypatch.setattr("radar.discord.time.sleep", slept.append)
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
