"""Discover and verify public job boards for the watchlist.

A board only counts as verified when there is evidence it belongs to the company:
Simplify links to it, its Greenhouse board/company name matches, or most of its
postings mention the company by name. Slugs that exist without evidence are left
for manual review instead of being trusted.

Usage:
  uv run python scripts/probe_boards.py [--seed PATH] [--out PATH]
      Probe the seed companies. Entries in the `discovered` group are kept.
  uv run python scripts/probe_boards.py --discover [--max-discovered N]
      Rebuild the `discovered` group from boards that Simplify links to for listings
      passing the radar's filters. Seed entries are kept.
"""

import argparse
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
import yaml

from radar.ats import Board, parse_ats_url
from radar.config import Settings, load_settings
from radar.filters import Match, PostingFilter
from radar.net import NetworkError, make_client, request_with_retries
from radar.sources.simplify import SimplifyFeedError, fetch_listings, parse_listings

log = logging.getLogger("probe")

DEFAULT_SEED = Path("config/watchlist_seed.yaml")
DEFAULT_OUT = Path("config/watchlist.yaml")
DESCRIPTION_SAMPLE = 10
MENTION_RATIO = 0.5
LEVER_MIN_INTERVAL = 1.05
DEFAULT_MIN_INTERVAL = 0.25
DEFAULT_MAX_DISCOVERED = 150
DISCOVERED_GROUP = "discovered"


class Status(StrEnum):
    VERIFIED = "verified"
    NEEDS_REVIEW = "needs_review"
    FEED_ONLY = "feed_only"
    PROBE_FAILED = "probe_failed"


class ProbeError(Exception):
    pass


@dataclass(frozen=True)
class SeedCompany:
    name: str
    group: str
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


@dataclass(frozen=True)
class Candidate:
    board: Board
    slug: str
    from_simplify: bool


@dataclass
class Snapshot:
    """What a board API returned. Descriptions stay in memory and are never written out."""

    board: Board
    slug: str
    open_jobs: int
    board_name: str | None = None
    job_company_names: list[str] = field(default_factory=list)
    descriptions: list[str] = field(default_factory=list)


@dataclass
class Result:
    company: SeedCompany
    status: Status
    board: Board | None = None
    slug: str | None = None
    open_jobs: int | None = None
    evidence: list[str] = field(default_factory=list)
    others: list[dict[str, Any]] = field(default_factory=list)
    note: str | None = None

    def to_yaml(self) -> dict[str, Any]:
        entry: dict[str, Any] = {"name": self.company.name, "group": self.company.group}
        entry["status"] = str(self.status)
        if self.board is not None:
            entry.update(board=str(self.board), slug=self.slug, open_jobs=self.open_jobs)
            entry["evidence"] = self.evidence
        if self.others:
            entry["other_boards" if self.status is Status.VERIFIED else "candidates"] = self.others
        if self.note:
            entry["note"] = self.note
        return entry


def normalize_name(name: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", name.casefold()).split())


def names_match(found: str, company: SeedCompany) -> bool:
    normalized = normalize_name(found)
    return any(
        normalized == expected or normalized.startswith(f"{expected} ")
        for expected in (normalize_name(name) for name in company.names)
    )


def board_from_url(url: str) -> tuple[Board, str] | None:
    """Extract (board, slug) from a job URL hosted on a supported ATS."""
    ref = parse_ats_url(url)
    return (ref.board, ref.slug) if ref and ref.slug else None


def simplify_evidence(records: Iterable[Any]) -> dict[str, set[tuple[Board, str]]]:
    """Map normalized company name -> boards that Simplify's own links point at."""
    found: dict[str, set[tuple[Board, str]]] = defaultdict(set)
    for record in records:
        if not isinstance(record, dict):
            continue
        company, url = record.get("company_name"), record.get("url")
        if not isinstance(company, str) or not isinstance(url, str):
            continue
        board = board_from_url(url)
        if board is not None:
            found[normalize_name(company)].add(board)
    return found


def candidates_for(
    company: SeedCompany, evidence: dict[str, set[tuple[Board, str]]]
) -> list[Candidate]:
    from_simplify = sorted(
        {pair for name in company.names for pair in evidence.get(normalize_name(name), set())}
    )
    candidates = [Candidate(board, slug, True) for board, slug in from_simplify]
    seen = {(candidate.board, candidate.slug) for candidate in candidates}

    for name in company.names:
        lower = name.casefold()
        for slug in (re.sub(r"[^a-z0-9]", "", lower), re.sub(r"[^a-z0-9]+", "-", lower).strip("-")):
            for board in Board:
                if slug and (board, slug) not in seen:
                    seen.add((board, slug))
                    candidates.append(Candidate(board, slug, False))
    return candidates


class BoardClient:
    def __init__(self, client: httpx.Client, sleep: Callable[[float], None] = time.sleep):
        self._client = client
        self._sleep = sleep
        self._last_request: dict[Board, float] = {}

    def snapshot(self, board: Board, slug: str) -> Snapshot | None:
        """Return None when the board does not exist; raise ProbeError when we can't tell."""
        match board:
            case Board.GREENHOUSE:
                return self._greenhouse(slug)
            case Board.LEVER:
                return self._lever(slug)
            case Board.ASHBY:
                return self._ashby(slug)

    def _get(self, board: Board, url: str, label: str, **params: str) -> Any | None:
        interval = LEVER_MIN_INTERVAL if board is Board.LEVER else DEFAULT_MIN_INTERVAL
        elapsed = time.monotonic() - self._last_request.get(board, 0.0)
        if elapsed < interval:
            self._sleep(interval - elapsed)
        try:
            response = request_with_retries(
                self._client, "GET", url, label=label, params=params, sleep=self._sleep
            )
        except NetworkError as exc:
            raise ProbeError(str(exc)) from exc
        finally:
            self._last_request[board] = time.monotonic()
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise ProbeError(f"{label}: HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as exc:
            raise ProbeError(f"{label}: response is not JSON") from exc

    def _greenhouse(self, slug: str) -> Snapshot | None:
        base = f"https://boards-api.greenhouse.io/v1/boards/{slug}"
        board = self._get(Board.GREENHOUSE, base, f"greenhouse/{slug}")
        if board is None:
            return None
        jobs_payload = self._get(Board.GREENHOUSE, f"{base}/jobs", f"greenhouse/{slug} jobs") or {}
        jobs = jobs_payload.get("jobs") or []
        return Snapshot(
            board=Board.GREENHOUSE,
            slug=slug,
            open_jobs=len(jobs),
            board_name=board.get("name"),
            job_company_names=[job.get("company_name") or "" for job in jobs],
        )

    def _lever(self, slug: str) -> Snapshot | None:
        url = f"https://api.lever.co/v0/postings/{slug}"
        postings = self._get(Board.LEVER, url, f"lever/{slug}", mode="json")
        if postings is None:
            return None
        if not isinstance(postings, list):
            raise ProbeError(f"lever/{slug}: expected a list")
        descriptions = [
            " ".join(
                str(posting.get(key) or "")
                for key in ("openingPlain", "descriptionPlain", "additionalPlain")
            )
            for posting in postings[:DESCRIPTION_SAMPLE]
        ]
        return Snapshot(Board.LEVER, slug, len(postings), descriptions=descriptions)

    def _ashby(self, slug: str) -> Snapshot | None:
        url = f"https://api.ashbyhq.com/posting-api/job-board/{slug}"
        payload = self._get(Board.ASHBY, url, f"ashby/{slug}")
        if payload is None:
            return None
        jobs = [job for job in payload.get("jobs") or [] if job.get("isListed", True)]
        descriptions = [str(job.get("descriptionPlain") or "") for job in jobs[:DESCRIPTION_SAMPLE]]
        return Snapshot(Board.ASHBY, slug, len(jobs), descriptions=descriptions)


def evidence_for(company: SeedCompany, snapshot: Snapshot, from_simplify: bool) -> list[str]:
    evidence: list[str] = []
    if from_simplify:
        evidence.append("simplify_url")
    if snapshot.board_name and names_match(snapshot.board_name, company):
        evidence.append("board_name")
    named = [name for name in snapshot.job_company_names if name]
    if named and sum(names_match(name, company) for name in named) / len(named) >= MENTION_RATIO:
        evidence.append("job_company_name")
    if snapshot.descriptions:
        pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(name) for name in company.names) + r")\b"
        )
        mentions = sum(bool(pattern.search(text)) for text in snapshot.descriptions)
        if mentions / len(snapshot.descriptions) >= MENTION_RATIO:
            evidence.append("description_mentions")
    return evidence


def probe_company(company: SeedCompany, candidates: list[Candidate], boards: BoardClient) -> Result:
    verified: list[tuple[Snapshot, list[str]]] = []
    unconfirmed: list[Snapshot] = []
    failures: list[str] = []

    for candidate in candidates:
        try:
            snapshot = boards.snapshot(candidate.board, candidate.slug)
        except ProbeError as exc:
            failures.append(str(exc))
            continue
        if snapshot is None:
            continue
        evidence = evidence_for(company, snapshot, candidate.from_simplify)
        if evidence:
            verified.append((snapshot, evidence))
        else:
            unconfirmed.append(snapshot)

    if verified:
        verified.sort(key=lambda item: item[0].open_jobs, reverse=True)
        (best, evidence), rest = verified[0], verified[1:]
        return Result(
            company,
            Status.VERIFIED,
            best.board,
            best.slug,
            best.open_jobs,
            evidence,
            others=[_describe(snap, ev) for snap, ev in rest],
        )
    if failures:
        return Result(company, Status.PROBE_FAILED, note="; ".join(failures))
    if unconfirmed:
        return Result(
            company,
            Status.NEEDS_REVIEW,
            others=[_describe(snap, []) for snap in unconfirmed],
            note="a board exists under this slug but nothing ties it to the company",
        )
    return Result(company, Status.FEED_ONLY, note="no public Greenhouse, Lever or Ashby board")


def _describe(snapshot: Snapshot, evidence: list[str]) -> dict[str, Any]:
    described: dict[str, Any] = {
        "board": str(snapshot.board),
        "slug": snapshot.slug,
        "open_jobs": snapshot.open_jobs,
    }
    if snapshot.board_name:
        described["board_name"] = snapshot.board_name
    if evidence:
        described["evidence"] = evidence
    return described


def _summary(result: Result) -> str:
    if result.board is None:
        return ""
    return f"{result.board}/{result.slug} ({result.open_jobs} jobs) {result.evidence}"


def load_seed(path: Path, excluded: list[str]) -> list[SeedCompany]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    excluded_names = {normalize_name(name) for name in excluded}
    companies: list[SeedCompany] = []
    for group, entries in raw.items():
        for entry in entries or []:
            company = SeedCompany(entry["name"], group, tuple(entry.get("aliases") or ()))
            if normalize_name(company.name) in excluded_names:
                log.warning("skipping %s: listed in excluded_companies", company.name)
                continue
            companies.append(company)
    return companies


@dataclass(frozen=True)
class DiscoveryCandidate:
    board: Board
    slug: str
    company: str
    listings: int


def discovery_candidates(
    records: list[Any], settings: Settings, known: set[tuple[str, str]]
) -> list[DiscoveryCandidate]:
    """Boards Simplify links to for listings that pass the radar's filters, most listings first.

    Age is ignored and inactive listings count: a board that hosted relevant internships
    before is likely to host the next ones. Excluded companies never match the filters.
    """
    posting_filter = PostingFilter(
        settings.filters,
        settings.locations,
        settings.excluded_companies,
        max_age=timedelta(days=3650),
    )
    now = datetime.now(UTC)
    counts: Counter[tuple[Board, str]] = Counter()
    names: dict[tuple[Board, str], Counter[str]] = defaultdict(Counter)
    for posting in parse_listings(records, max_invalid_ratio=1.0).postings:
        if not isinstance(
            posting_filter.evaluate(posting.model_copy(update={"active": True}), now), Match
        ):
            continue
        ref = parse_ats_url(str(posting.url))
        if ref is None or not ref.slug or (str(ref.board), ref.slug) in known:
            continue
        counts[(ref.board, ref.slug)] += 1
        names[(ref.board, ref.slug)][posting.company] += 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    return [
        DiscoveryCandidate(board, slug, names[(board, slug)].most_common(1)[0][0], count)
        for (board, slug), count in ranked
    ]


def discover(
    candidates: list[DiscoveryCandidate], boards: BoardClient, limit: int
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Verify candidates in rank order until `limit` boards answer. Simplify linking to the
    board is the evidence; the API call proves the board still exists."""
    entries: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for candidate in candidates:
        if len(entries) >= limit:
            break
        try:
            snapshot = boards.snapshot(candidate.board, candidate.slug)
        except ProbeError:
            skipped["probe_failed"] += 1
            continue
        if snapshot is None:
            skipped["not_found"] += 1
            continue
        entries.append(
            {
                "name": candidate.company,
                "group": DISCOVERED_GROUP,
                "status": str(Status.VERIFIED),
                "board": str(candidate.board),
                "slug": candidate.slug,
                "open_jobs": snapshot.open_jobs,
                "evidence": ["simplify_url"],
                "simplify_listings": candidate.listings,
            }
        )
    return entries, skipped


def read_watchlist(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("companies") or []


def known_boards(entries: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """Boards already verified. Unproven `needs_review` candidates stay discoverable, since a
    Simplify link is exactly the evidence they were missing."""
    known = set()
    for entry in entries:
        if entry.get("status") != str(Status.VERIFIED):
            continue
        if entry.get("board") and entry.get("slug"):
            known.add((entry["board"], entry["slug"]))
        for other in entry.get("other_boards") or []:
            known.add((other["board"], other["slug"]))
    return known


def write_watchlist(path: Path, entries: list[dict[str, Any]]) -> None:
    header = (
        "# Generated by scripts/probe_boards.py. Do not edit by hand: change\n"
        "# config/watchlist_seed.yaml and re-run the probe, or re-run it with --discover.\n"
    )
    body = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "companies": entries,
    }
    path.write_text(header + yaml.safe_dump(body, sort_keys=False, width=100), encoding="utf-8")


def _quiet_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run_discovery(out: Path, limit: int) -> int:
    settings = load_settings()
    existing = read_watchlist(out)
    kept = [entry for entry in existing if entry.get("group") != DISCOVERED_GROUP]
    with make_client() as client:
        try:
            records = fetch_listings(client, settings.sources.simplify.url)
        except SimplifyFeedError as exc:
            log.error("cannot discover without the Simplify feed: %s", exc)
            return 1
        candidates = discovery_candidates(records, settings, known_boards(kept))
        entries, skipped = discover(candidates, BoardClient(client), limit)

    write_watchlist(out, kept + entries)
    by_board = Counter(entry["board"] for entry in entries)
    log.info("candidates=%d discovered=%d skipped[%s] by_board[%s]", len(candidates), len(entries),
             ", ".join(f"{k}={v}" for k, v in sorted(skipped.items())),
             ", ".join(f"{k}={v}" for k, v in sorted(by_board.items())))  # fmt: skip
    for entry in entries[:15]:
        log.info("  %-35s %s/%s listings=%d", entry["name"][:35], entry["board"], entry["slug"],
                 entry["simplify_listings"])  # fmt: skip
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--max-discovered", type=int, default=DEFAULT_MAX_DISCOVERED)
    args = parser.parse_args(argv)
    _quiet_logging()
    if args.discover:
        return run_discovery(args.out, args.max_discovered)

    settings = load_settings()
    companies = load_seed(args.seed, settings.excluded_companies)

    with make_client() as client:
        try:
            evidence = simplify_evidence(fetch_listings(client, settings.sources.simplify.url))
        except SimplifyFeedError as exc:
            log.warning("continuing without Simplify evidence: %s", exc)
            evidence = {}

        boards = BoardClient(client)
        results: list[Result] = []
        for company in companies:
            result = probe_company(company, candidates_for(company, evidence), boards)
            results.append(result)
            log.info("%-12s %-13s %s", company.name, result.status, _summary(result))

    discovered = [e for e in read_watchlist(args.out) if e.get("group") == DISCOVERED_GROUP]
    write_watchlist(args.out, [result.to_yaml() for result in results] + discovered)
    failed = sum(result.status is Status.PROBE_FAILED for result in results)
    log.info("wrote %s: %d companies, %d probe failures", args.out, len(results), failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
