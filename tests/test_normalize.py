from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
import respx

from radar.net import make_client
from radar.sources.simplify import (
    SimplifyFeedError,
    fetch_listings,
    load_postings,
    parse_listings,
)
from tests.conftest import FEED_URL


def test_parses_recorded_feed_and_skips_hidden(raw_listings: list[dict[str, Any]]) -> None:
    result = parse_listings(raw_listings, max_invalid_ratio=0.05)

    assert result.total == 13
    assert result.hidden == 1
    assert result.invalid == 0
    assert len(result.postings) == 12


def test_maps_fields_into_shared_model(raw_listings: list[dict[str, Any]]) -> None:
    raw = raw_listings[0]
    posting = parse_listings([raw], max_invalid_ratio=0).postings[0]

    assert posting.source == "simplify"
    assert posting.source_id == raw["id"]
    assert posting.key == f"simplify:{raw['id']}"
    assert posting.company == "Andreessen Horowitz"
    assert posting.title == "Research Intern - Frontier"
    assert posting.locations == ["NYC"]
    assert posting.terms == ["Summer 2027"]
    assert posting.category == "AI/ML/Data"
    assert posting.posted_at == datetime.fromtimestamp(raw["date_posted"], tz=UTC)
    assert posting.posted_at.tzinfo is not None
    assert str(posting.url) == raw["url"]
    assert posting.description == ""


def test_ignores_unknown_upstream_fields(raw_listings: list[dict[str, Any]]) -> None:
    raw = {**raw_listings[0], "brand_new_field": {"nested": True}}

    assert len(parse_listings([raw], max_invalid_ratio=0).postings) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        {"title": ""},
        {"url": "not a url"},
        {"date_posted": "yesterday"},
        {"locations": "NYC"},
    ],
)
def test_counts_invalid_records_without_crashing(
    raw_listings: list[dict[str, Any]], mutation: dict[str, Any]
) -> None:
    records = [*raw_listings, {**raw_listings[0], **mutation}]

    result = parse_listings(records, max_invalid_ratio=0.5)

    assert result.invalid == 1
    assert len(result.postings) == 12


def test_missing_required_field_counts_as_invalid(raw_listings: list[dict[str, Any]]) -> None:
    broken = {key: value for key, value in raw_listings[0].items() if key != "date_posted"}

    assert parse_listings([*raw_listings, broken], max_invalid_ratio=0.5).invalid == 1


def test_too_many_invalid_records_signals_schema_change(raw_listings: list[dict[str, Any]]) -> None:
    broken = [{"unexpected": "shape"}] * 5

    with pytest.raises(SimplifyFeedError, match="schema may have changed"):
        parse_listings([*raw_listings, *broken], max_invalid_ratio=0.05)


def test_empty_feed_is_an_error() -> None:
    with pytest.raises(SimplifyFeedError, match="zero listings"):
        parse_listings([], max_invalid_ratio=0.05)


@respx.mock
def test_fetch_returns_records(raw_listings: list[dict[str, Any]]) -> None:
    respx.get(FEED_URL).mock(return_value=httpx.Response(200, json=raw_listings))

    with make_client() as client:
        result = load_postings(client, FEED_URL, max_invalid_ratio=0.05)

    assert len(result.postings) == 12


@respx.mock
def test_fetch_retries_server_errors_then_succeeds(
    raw_listings: list[dict[str, Any]], no_real_sleep: list[float]
) -> None:
    route = respx.get(FEED_URL).mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json=raw_listings)]
    )

    with make_client() as client:
        records = fetch_listings(client, FEED_URL)

    assert route.call_count == 2
    assert len(records) == 13
    assert no_real_sleep == [1.0]


@respx.mock
def test_fetch_gives_up_after_retries() -> None:
    respx.get(FEED_URL).mock(return_value=httpx.Response(502))

    with make_client() as client, pytest.raises(SimplifyFeedError, match="after 3 attempts"):
        fetch_listings(client, FEED_URL)


@respx.mock
def test_fetch_treats_timeouts_as_retryable() -> None:
    respx.get(FEED_URL).mock(side_effect=httpx.ReadTimeout("slow"))

    with make_client() as client, pytest.raises(SimplifyFeedError, match="timed out"):
        fetch_listings(client, FEED_URL)


@respx.mock
@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(404), "HTTP 404"),
        (httpx.Response(200, text="<html>"), "not valid JSON"),
        (httpx.Response(200, json={"listings": []}), "expected a list"),
    ],
)
def test_fetch_rejects_unexpected_responses(response: httpx.Response, message: str) -> None:
    respx.get(FEED_URL).mock(return_value=response)

    with make_client() as client, pytest.raises(SimplifyFeedError, match=message):
        fetch_listings(client, FEED_URL)
