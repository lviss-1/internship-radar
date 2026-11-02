import json

import httpx
import pytest
import respx

from radar.discord import (
    SUPPRESS_NOTIFICATIONS,
    TITLE_LIMIT,
    DiscordError,
    DiscordWebhook,
    match_payload,
    text_payload,
)
from radar.filters import LocationKind, Match, Tag
from radar.net import make_client
from tests.conftest import WEBHOOK_URL, make_posting

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


def test_match_payload_for_local_role() -> None:
    match = Match(make_posting(), LocationKind.IN_PERSON)

    payload = match_payload(match)

    embed = payload["embeds"][0]
    assert payload["flags"] == 0
    assert payload["allowed_mentions"] == {"parse": []}
    assert embed["title"] == "Example Co — Software Engineer Intern"
    assert embed["url"] == "https://example.com/jobs/1"
    assert {field["name"] for field in embed["fields"]} == {"Location", "Fit", "Season", "Posted"}


def test_match_payload_for_relocation_role_is_silent_and_tagged() -> None:
    match = Match(
        make_posting(locations=["Austin, TX"], terms=["N/A"]),
        LocationKind.OTHER_US,
        (Tag.SEASON_UNCONFIRMED, Tag.RELOCATION_UNKNOWN),
    )

    payload = match_payload(match)

    tags = next(f for f in payload["embeds"][0]["fields"] if f["name"] == "Tags")
    assert payload["flags"] == SUPPRESS_NOTIFICATIONS
    assert tags["value"] == "season unconfirmed, relocation unknown"


def test_match_payload_truncates_long_titles() -> None:
    match = Match(make_posting(title="Intern " * 100), LocationKind.IN_PERSON)

    assert len(match_payload(match)["embeds"][0]["title"]) == TITLE_LIMIT
