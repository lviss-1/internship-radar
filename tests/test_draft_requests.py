from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import httpx2
import pytest
import respx

from radar.config import Settings
from radar.feedback import DISCORD_API, FeedbackConfig
from radar.main import RunSummary
from radar.state import FileStateStore
from tests.conftest import FEED_URL, NOW, WEBHOOK_URL, FakeClaude, anthropic_fixture
from tests.test_state import RecordingHook, fake_model, offers_drafts, run_once

CHANNEL = "555"
WEBHOOK_ID = "123"
MESSAGES = f"{DISCORD_API}/channels/{CHANNEL}/messages"
FEEDBACK = FeedbackConfig(bot_token="bot-token", webhook_url=WEBHOOK_URL)
DRAFTABLE = {"Andreessen Horowitz", "KnowBe4"}
WRITING_HAND = "\N{WRITING HAND}️"


class Channel:
    """The matches channel as the bot sees it: every card posted so far, newest first,
    with a ✍️ reaction on the ones whose company is in `reacted`."""

    def __init__(self, hook: RecordingHook) -> None:
        self.hook = hook
        self.reacted: set[str] = set()

    def messages(self) -> list[dict[str, Any]]:
        messages = []
        for index, payload in enumerate(self.hook.sent):
            embed = payload["embeds"][0]
            company = embed["title"].split(" — ")[0]
            reactions = [{"emoji": {"name": WRITING_HAND}, "count": 1}]
            messages.append(
                {
                    "id": str(1000 + index),
                    "timestamp": NOW.isoformat(),
                    "webhook_id": WEBHOOK_ID,
                    "embeds": [embed],
                    "reactions": reactions if company in self.reacted else [],
                }
            )
        return list(reversed(messages))


@contextmanager
def discord(raw_listings: list[dict[str, Any]], channel: Channel) -> Iterator[None]:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(FEED_URL).mock(return_value=httpx.Response(200, json=raw_listings))
        mock.get(WEBHOOK_URL).mock(
            return_value=httpx.Response(200, json={"channel_id": CHANNEL, "id": WEBHOOK_ID})
        )
        mock.get(MESSAGES).mock(
            side_effect=lambda request: httpx.Response(200, json=channel.messages())
        )
        mock.route().mock(return_value=httpx.Response(404))
        yield


def drafts_sent(hook: RecordingHook) -> list[str]:
    return [
        p["embeds"][0]["title"].removeprefix("Drafts: ").split(" — ")[0]
        for p in hook.sent
        if p["embeds"][0]["title"].startswith("Drafts: ")
    ]


def sonnet_calls(claude: FakeClaude) -> int:
    return sum(body["model"] == "claude-sonnet-5-5" for body in claude.requests)


def drafts_model(respond: Callable[[], httpx2.Response]) -> Callable[..., httpx2.Response]:
    def model(body: dict[str, Any]) -> httpx2.Response:
        return respond() if body["model"] == "claude-sonnet-5-5" else fake_model(body)

    return model


Runner = Callable[..., RunSummary]


@pytest.fixture
def radar(
    settings: Settings, tmp_path: Path, raw_listings: list[dict[str, Any]]
) -> Iterator[tuple[Runner, Channel, RecordingHook, FileStateStore]]:
    """A first run posts the cards; each later call runs again with the channel as it is."""
    store = FileStateStore(tmp_path)
    hook = RecordingHook()
    channel = Channel(hook)

    def again(claude: FakeClaude, run_settings: Settings = settings, **kwargs: Any) -> RunSummary:
        with discord(raw_listings, channel):
            return run_once(run_settings, store, hook, claude=claude, feedback=FEEDBACK, **kwargs)

    again(FakeClaude(fake_model))
    yield again, channel, hook, store


def test_first_run_offers_drafts_on_high_scores_and_drafts_nothing(radar: Any) -> None:
    _, _, hook, store = radar

    offered = {c["embeds"][0]["title"].split(" — ")[0] for c in hook.cards() if offers_drafts(c)}
    assert offered == DRAFTABLE
    assert drafts_sent(hook) == []
    assert store.load().drafted == set()


def test_reacted_draftable_cards_get_drafts_exactly_once(radar: Any) -> None:
    again, channel, hook, store = radar
    claude = FakeClaude(fake_model)
    channel.reacted = {"Andreessen Horowitz", "KnowBe4", "Microsoft", "Entrust"}

    first = again(claude)
    second = again(claude)

    assert sorted(drafts_sent(hook)) == sorted(DRAFTABLE)
    assert (first.draft_requests, first.drafted, first.dropped_edits) == (2, 2, 2)
    assert (second.draft_requests, second.drafted) == (0, 0)
    assert sonnet_calls(claude) == 2
    assert len(store.load().drafted) == 2


def test_lasting_model_failure_is_not_retried(radar: Any) -> None:
    again, channel, hook, store = radar
    claude = FakeClaude(
        drafts_model(lambda: httpx2.Response(200, json=anthropic_fixture("refusal_message.json")))
    )
    channel.reacted = {"Andreessen Horowitz"}

    first = again(claude)
    again(claude)

    assert first.draft_failures == {"refused": 1}
    assert drafts_sent(hook) == []
    assert sonnet_calls(claude) == 1
    assert len(store.load().drafted) == 1


def test_transient_model_failure_retries_next_run(radar: Any) -> None:
    again, channel, hook, store = radar
    overloaded = {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
    channel.reacted = {"Andreessen Horowitz"}

    failed = again(FakeClaude(drafts_model(lambda: httpx2.Response(529, json=overloaded))))
    assert failed.draft_failures == {"api_http_529": 1}
    assert store.load().drafted == set()

    retried = again(FakeClaude(fake_model))
    assert retried.drafted == 1
    assert drafts_sent(hook) == ["Andreessen Horowitz"]


def test_failed_drafts_post_is_released_for_the_next_run(radar: Any) -> None:
    again, channel, hook, store = radar
    channel.reacted = {"Andreessen Horowitz"}
    hook.fail_titles = {"Drafts: "}

    failed = again(FakeClaude(fake_model))
    assert failed.draft_failures == {"discord": 1}
    assert store.load().drafted == set()

    hook.fail_titles = set()
    assert again(FakeClaude(fake_model)).drafted == 1
    assert drafts_sent(hook) == ["Andreessen Horowitz"]


def test_drafts_per_run_are_capped_and_the_rest_wait(radar: Any, settings: Settings) -> None:
    again, channel, hook, _ = radar
    capped = settings.model_copy(
        update={"delivery": settings.delivery.model_copy(update={"max_drafts_per_run": 1})}
    )
    channel.reacted = set(DRAFTABLE)

    first = again(FakeClaude(fake_model), capped)
    second = again(FakeClaude(fake_model), capped)

    assert (first.drafted, first.drafts_deferred) == (1, 1)
    assert (second.drafted, second.drafts_deferred) == (1, 0)
    assert sorted(drafts_sent(hook)) == sorted(DRAFTABLE)
