import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from radar.net import NetworkError, request_with_retries
from radar.normalize import Posting

log = logging.getLogger(__name__)


class SimplifyFeedError(Exception):
    pass


class SimplifyListing(BaseModel):
    """One record of listings.json. Extra fields are ignored so upstream additions are safe."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: str
    locations: list[str]
    terms: list[str]
    date_posted: int
    active: bool
    is_visible: bool
    category: str | None = None


@dataclass(frozen=True)
class ParseResult:
    postings: list[Posting]
    total: int
    invalid: int
    hidden: int


def fetch_listings(client: httpx.Client, url: str) -> list[Any]:
    try:
        response = request_with_retries(client, "GET", url, label="simplify feed")
    except NetworkError as exc:
        raise SimplifyFeedError(str(exc)) from exc
    if response.status_code != 200:
        raise SimplifyFeedError(f"simplify feed: HTTP {response.status_code}")
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise SimplifyFeedError("simplify feed: response is not valid JSON") from exc
    if not isinstance(payload, list):
        raise SimplifyFeedError(f"simplify feed: expected a list, got {type(payload).__name__}")
    return payload


def to_posting(listing: SimplifyListing) -> Posting:
    return Posting(
        source="simplify",
        source_id=listing.id,
        company=listing.company_name.strip(),
        title=listing.title.strip(),
        locations=[location.strip() for location in listing.locations if location.strip()],
        url=listing.url.strip(),
        posted_at=datetime.fromtimestamp(listing.date_posted, tz=UTC),
        terms=[term.strip() for term in listing.terms],
        category=listing.category,
        active=listing.active,
    )


def parse_listings(records: list[Any], max_invalid_ratio: float) -> ParseResult:
    """Validate and normalize raw records, skipping bad ones.

    Raises SimplifyFeedError when the feed is empty or too many records fail validation,
    which usually means the upstream schema changed.
    """
    if not records:
        raise SimplifyFeedError("simplify feed: zero listings returned")

    postings: list[Posting] = []
    invalid = 0
    hidden = 0
    for record in records:
        try:
            listing = SimplifyListing.model_validate(record)
            if not listing.is_visible:
                hidden += 1
                continue
            postings.append(to_posting(listing))
        except ValidationError:
            invalid += 1

    ratio = invalid / len(records)
    if ratio > max_invalid_ratio:
        raise SimplifyFeedError(
            f"simplify feed: {invalid}/{len(records)} records failed validation "
            f"({ratio:.1%} > {max_invalid_ratio:.1%}); upstream schema may have changed"
        )
    if invalid:
        log.warning("simplify: skipped %d invalid records", invalid)
    return ParseResult(postings=postings, total=len(records), invalid=invalid, hidden=hidden)


def load_postings(client: httpx.Client, url: str, max_invalid_ratio: float) -> ParseResult:
    return parse_listings(fetch_listings(client, url), max_invalid_ratio)
