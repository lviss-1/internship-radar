"""Read the verified watchlist boards (Greenhouse, Lever, Ashby) into Postings.

Descriptions pass through here and must never be logged.
"""

import html
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from radar.ats import Board
from radar.net import NetworkError, request_with_retries
from radar.normalize import Posting, Source, infer_terms
from radar.sources.descriptions import html_to_text, lever_text, truncate_description

BOARD_INTERVAL = 0.2
LEVER_MIN_INTERVAL = 1.05

Wants = Callable[[Source, str, datetime, frozenset[str]], bool]


class WatchlistError(Exception):
    pass


class BoardError(Exception):
    pass


@dataclass(frozen=True)
class WatchlistBoard:
    company: str
    board: Board
    slug: str

    @property
    def key(self) -> str:
        return f"{self.board}/{self.slug}"


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _OtherBoard(_Lenient):
    board: Board
    slug: str
    evidence: list[str] = []


class _Company(_Lenient):
    name: str
    status: str
    board: Board | None = None
    slug: str | None = None
    other_boards: list[_OtherBoard] = []


class _Watchlist(_Lenient):
    generated_at: datetime | None = None
    companies: list[_Company]


@dataclass(frozen=True)
class LoadedWatchlist:
    boards: list[WatchlistBoard]
    generated_at: datetime | None


def _normalized(name: str) -> str:
    return " ".join(name.casefold().split())


def load_watchlist(path: Path, excluded_companies: list[str]) -> LoadedWatchlist:
    """Verified primary boards plus extra boards that carry their own evidence (campus boards).

    Boards the probe could only find, but not tie to the company, are never polled.
    """
    try:
        watchlist = _Watchlist.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise WatchlistError(f"watchlist {path} is missing or invalid") from exc

    excluded = {_normalized(name) for name in excluded_companies}
    boards: list[WatchlistBoard] = []
    for company in watchlist.companies:
        if company.status != "verified" or _normalized(company.name) in excluded:
            continue
        if company.board and company.slug:
            boards.append(WatchlistBoard(company.name, company.board, company.slug))
        boards.extend(
            WatchlistBoard(company.name, other.board, other.slug)
            for other in company.other_boards
            if other.evidence
        )
    return LoadedWatchlist(boards, watchlist.generated_at)


@dataclass
class BoardResult:
    postings: list[Posting] = field(default_factory=list)
    published: dict[str, datetime] = field(default_factory=dict)
    polled: list[str] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _ashby_locations(job: dict[str, Any]) -> list[str]:
    places = [job.get("location") or ""]
    places += [item.get("location") or "" for item in job.get("secondaryLocations") or []]
    address = (job.get("address") or {}).get("postalAddress") or {}
    country = address.get("addressCountry") or ""
    parts = [address.get("addressLocality"), address.get("addressRegion"), country]
    places.append(", ".join(part for part in parts if part))
    if job.get("workplaceType") == "Remote":
        places.append(f"Remote in {country}" if country else "Remote")
    return [place for place in places if place]


class BoardSource:
    def __init__(
        self,
        client: httpx.Client,
        max_description_chars: int,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._client = client
        self._max_chars = max_description_chars
        self._sleep = sleep or time.sleep
        self._last_lever_request = 0.0

    def fetch(self, boards: list[WatchlistBoard], wants: Wants) -> BoardResult:
        """Poll every board. `wants` decides, from cheap fields only, which jobs are worth a
        full Posting (and, on Greenhouse, a detail request)."""
        result = BoardResult()
        fetchers = {Board.GREENHOUSE: self._greenhouse, Board.LEVER: self._lever,
                    Board.ASHBY: self._ashby}  # fmt: skip
        for index, board in enumerate(boards):
            if index:
                self._sleep(BOARD_INTERVAL)
            try:
                fetchers[board.board](board, wants, result)
            except BoardError as exc:
                result.failures[board.key] = str(exc)
                continue
            except NetworkError as exc:
                result.failures[board.key] = exc.reason
                continue
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                result.failures[board.key] = f"unexpected response ({type(exc).__name__})"
                continue
            result.polled.append(board.key)
        return result

    def _get_json(self, url: str, label: str, **params: str) -> Any:
        response = request_with_retries(
            self._client, "GET", url, label=label, params=params, sleep=self._sleep
        )
        if response.status_code != 200:
            raise BoardError(f"HTTP {response.status_code}")
        return response.json()

    def _add(self, result: BoardResult, **fields: Any) -> None:
        try:
            result.postings.append(Posting(**fields))
        except ValidationError:
            return

    def _greenhouse(self, board: WatchlistBoard, wants: Wants, result: BoardResult) -> None:
        base = f"https://boards-api.greenhouse.io/v1/boards/{board.slug}/jobs"
        for job in self._get_json(base, "greenhouse board").get("jobs") or []:
            job_id, title = str(job["id"]), str(job.get("title") or "")
            posted_at = _parse_time(job.get("first_published") or job.get("updated_at"))
            if posted_at is None:
                continue
            key = f"greenhouse:{job_id}"
            result.published[key] = posted_at
            if not wants("greenhouse", title, posted_at, frozenset({key})):
                continue

            locations = [(job.get("location") or {}).get("name") or ""]
            description = ""
            try:
                detail = self._get_json(f"{base}/{job_id}", "greenhouse job")
            except (BoardError, NetworkError, ValueError):
                detail = {}
            for office in detail.get("offices") or []:
                locations.append(office.get("location") or office.get("name") or "")
            if detail.get("content"):
                text = html_to_text(html.unescape(detail["content"]))
                description = truncate_description(text, self._max_chars)
            self._add(
                result,
                source="greenhouse",
                source_id=job_id,
                company=board.company,
                title=title,
                locations=[place for place in locations if place],
                url=job.get("absolute_url"),
                posted_at=posted_at,
                description=description,
                terms=infer_terms(title),
                board=board.key,
            )

    def _ashby(self, board: WatchlistBoard, wants: Wants, result: BoardResult) -> None:
        url = f"https://api.ashbyhq.com/posting-api/job-board/{board.slug}"
        for job in self._get_json(url, "ashby board").get("jobs") or []:
            if not job.get("isListed", True):
                continue
            job_id, title = str(job["id"]), str(job.get("title") or "")
            posted_at = _parse_time(job.get("publishedAt"))
            if posted_at is None:
                continue
            key = f"ashby:{job_id}"
            result.published[key] = posted_at
            if not wants("ashby", title, posted_at, frozenset({key})):
                continue
            text = job.get("descriptionPlain") or html_to_text(job.get("descriptionHtml") or "")
            self._add(
                result,
                source="ashby",
                source_id=job_id,
                company=board.company,
                title=title,
                locations=_ashby_locations(job),
                url=job.get("jobUrl") or f"https://jobs.ashbyhq.com/{board.slug}/{job_id}",
                posted_at=posted_at,
                description=truncate_description(text, self._max_chars),
                terms=infer_terms(title),
                board=board.key,
            )

    def _lever(self, board: WatchlistBoard, wants: Wants, result: BoardResult) -> None:
        elapsed = time.monotonic() - self._last_lever_request
        if elapsed < LEVER_MIN_INTERVAL:
            self._sleep(LEVER_MIN_INTERVAL - elapsed)
        try:
            url = f"https://api.lever.co/v0/postings/{board.slug}"
            postings = self._get_json(url, "lever board", mode="json")
        finally:
            self._last_lever_request = time.monotonic()
        if not isinstance(postings, list):
            raise BoardError("expected a list of postings")

        for job in postings:
            job_id, title = str(job["id"]), str(job.get("text") or "")
            created = job.get("createdAt")
            if not isinstance(created, int | float):
                continue
            posted_at = datetime.fromtimestamp(created / 1000, tz=UTC)
            key = f"lever:{job_id}"
            result.published[key] = posted_at
            if not wants("lever", title, posted_at, frozenset({key})):
                continue
            categories = job.get("categories") or {}
            locations = list(categories.get("allLocations") or [categories.get("location") or ""])
            if job.get("workplaceType") == "remote":
                country = job.get("country") or ""
                locations.append(f"Remote in {country}" if country else "Remote")
            self._add(
                result,
                source="lever",
                source_id=job_id,
                company=board.company,
                title=title,
                locations=[place for place in locations if place],
                url=job.get("hostedUrl"),
                posted_at=posted_at,
                description=truncate_description(lever_text(job), self._max_chars),
                terms=infer_terms(title),
                board=board.key,
            )
