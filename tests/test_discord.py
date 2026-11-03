import json

import httpx
import pytest
import respx

from radar.delivery import Decision, Evaluation
from radar.discord import (
    SUPPRESS_NOTIFICATIONS,
    TITLE_LIMIT,
    DiscordError,
    DiscordWebhook,
    drafts_payload,
    score_card_payload,
    text_payload,
)
from radar.drafts import DraftResult, ResumeEdit
from radar.filters import LocationKind, Match
from radar.net import make_client
from radar.scoring import ScoreResult
from tests.conftest import WEBHOOK_URL, make_posting

SCORE = ScoreResult(
    resume="swe",
    score=91,
    reasons=["Python services", "NYC office"],
    gaps=["No Kubernetes"],
    relocation_mentioned=False,
    eligibility_flags=["graduating 2027 only"],
)

SECRET_FRAGMENT = "test-token"


def hook(client: httpx.Client) -> DiscordWebhook:
    return DiscordWebhook(client, WEBHOOK_URL, label="DISCORD_WEBHOOK_MATCHES")


@pytest.mark.parametrize(
    "url",
    [
        "http://discord.com/api/webhooks/1/test-token",
        "https://evil.example/api/webhooks/1/test-token",
        "https://discord.com/channels/1/test-token",
    ],
)
def test_rejects_non_discord_urls_without_echoing_them(url: str) -> None:
    with make_client() as client, pytest.raises(DiscordError) as error:
        DiscordWebhook(client, url, label="DISCORD_WEBHOOK_MATCHES")

    assert SECRET_FRAGMENT not in str(error.value)


def test_repr_hides_url() -> None:
    with make_client() as client:
        assert SECRET_FRAGMENT not in repr(hook(client))


@respx.mock
def test_send_waits_for_confirmation_and_blocks_mentions() -> None:
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200, json={"id": "1"}))

    with make_client() as client:
        hook(client).send(text_payload("hello @everyone"))

    request = route.calls.last.request
    assert request.url.params["wait"] == "true"
    assert json.loads(request.content)["allowed_mentions"] == {"parse": []}


@respx.mock
def test_retries_after_rate_limit(no_real_sleep: list[float]) -> None:
    route = respx.post(WEBHOOK_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "2.5"}, json={"retry_after": 2.5}),
            httpx.Response(200, json={"id": "1"}),
        ]
    )

    with make_client() as client:
        hook(client).send(text_payload("hi"))

    assert route.call_count == 2
    assert no_real_sleep == [2.5]


@respx.mock
def test_waits_out_exhausted_bucket(no_real_sleep: list[float]) -> None:
    headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset-After": "1.25"}
    respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200, headers=headers, json={}))

    with make_client() as client:
        hook(client).send(text_payload("hi"))

    assert no_real_sleep == [1.25]


@respx.mock
@pytest.mark.parametrize("status", [400, 401, 404])
def test_client_errors_fail_without_leaking_url(status: int) -> None:
    respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(status))

    with make_client() as client, pytest.raises(DiscordError) as error:
        hook(client).send(text_payload("hi"))

    assert f"HTTP {status}" in str(error.value)
    assert SECRET_FRAGMENT not in str(error.value)


@respx.mock
def test_network_failure_does_not_leak_url() -> None:
    respx.post(WEBHOOK_URL).mock(side_effect=httpx.ConnectError(f"cannot reach {WEBHOOK_URL}"))

    with make_client() as client, pytest.raises(DiscordError) as error:
        hook(client).send(text_payload("hi"))

    assert SECRET_FRAGMENT not in str(error.value)
    assert SECRET_FRAGMENT not in str(error.value.__cause__)


def scored_evaluation(**overrides: object) -> Evaluation:
    fields: dict[str, object] = {
        "match": Match(make_posting(), LocationKind.IN_PERSON),
        "description_found": True,
        "score": SCORE,
    }
    fields.update(overrides)
    return Evaluation(**fields)  # type: ignore[arg-type]


def embed_size(embed: dict) -> int:
    return (
        len(embed["title"])
        + len(embed.get("description", ""))
        + len(embed["footer"]["text"])
        + sum(len(f["name"]) + len(f["value"]) for f in embed["fields"])
    )


def test_ping_mentions_only_the_configured_user() -> None:
    decision = Decision(post=True, silent=False, ping=True, tags=())

    payload = score_card_payload(scored_evaluation(), decision, "123456789")

    assert payload["content"] == "<@123456789>"
    assert payload["allowed_mentions"] == {"parse": [], "users": ["123456789"]}
    assert payload["flags"] == 0


def test_score_card_without_ping_mentions_nobody() -> None:
    decision = Decision(post=True, silent=True, ping=False, tags=("no relocation mentioned",))

    payload = score_card_payload(scored_evaluation(), decision, "123456789")

    names = [f["name"] for f in payload["embeds"][0]["fields"]]
    assert "content" not in payload
    assert payload["allowed_mentions"] == {"parse": [], "users": []}
    assert payload["flags"] == SUPPRESS_NOTIFICATIONS
    assert names == [
        "Score", "Fit", "Location", "Season", "Posted",
        "Why it fits", "Gaps", "Eligibility flags", "Tags",
    ]  # fmt: skip
    assert payload["embeds"][0]["fields"][0]["value"] == "91/100 · SWE resume"


def test_unscored_card_states_the_reason() -> None:
    item = scored_evaluation(score=None, unscored_reason="refused")

    payload = score_card_payload(item, Decision(post=True, tags=("unscored",)), None)

    fields = payload["embeds"][0]["fields"]
    assert fields[0]["value"] == "Unscored (refused)"
    assert "Why it fits" not in [f["name"] for f in fields]


def test_long_titles_are_truncated() -> None:
    item = scored_evaluation(
        match=Match(make_posting(title="Intern " * 100), LocationKind.IN_PERSON)
    )

    payload = score_card_payload(item, Decision(post=True), None)

    assert len(payload["embeds"][0]["title"]) == TITLE_LIMIT


def test_drafts_payload_fits_discord_limits_at_maximum_sizes() -> None:
    edits = [ResumeEdit(original="o" * 600, suggested="s" * 600, why="w" * 300)] * 5
    drafts = DraftResult(resume_edits=edits, cover_note="c" * 1500, referral_message="r" * 800)

    payload = drafts_payload(scored_evaluation(drafts=drafts))

    embed = payload["embeds"][0]
    assert embed_size(embed) <= 6000
    assert all(len(f["value"]) <= 1024 for f in embed["fields"])
    assert payload["flags"] == SUPPRESS_NOTIFICATIONS
    assert payload["allowed_mentions"] == {"parse": []}


def test_drafts_payload_says_when_no_edits_survived() -> None:
    drafts = DraftResult(resume_edits=[], cover_note="note", referral_message="hi")

    fields = drafts_payload(scored_evaluation(drafts=drafts))["embeds"][0]["fields"]

    assert fields[-1]["value"] == "No grounded edits suggested."
