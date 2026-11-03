"""Recognize job URLs hosted on the applicant tracking systems the radar reads directly."""

import re
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import parse_qs, urlsplit

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


class Board(StrEnum):
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    ASHBY = "ashby"


@dataclass(frozen=True)
class AtsRef:
    board: Board
    slug: str | None
    job_id: str | None

    @property
    def job_key(self) -> str | None:
        """The dedupe key the board source would give this job, e.g. "greenhouse:123".

        Greenhouse job IDs are unique across all boards, so a company-hosted link with only
        a `gh_jid` parameter still identifies the job.
        """
        return f"{self.board}:{self.job_id}" if self.job_id else None


def _uuid(segment: str | None) -> str | None:
    if segment is None:
        return None
    candidate = segment.lower()
    return candidate if _UUID.fullmatch(candidate) else None


def parse_ats_url(url: str) -> AtsRef | None:
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    segments = [segment for segment in parts.path.split("/") if segment]
    query = parse_qs(parts.query)

    if host.endswith("greenhouse.io") and "api" not in host:
        if segments and segments[0] == "embed":
            slug = query.get("for", [None])[0]
            job_id = query.get("token", [None])[0]
        else:
            slug = segments[0] if segments else None
            job_id = segments[2] if len(segments) >= 3 and segments[1] == "jobs" else None
        job_id = job_id if job_id and job_id.isdigit() else None
        return AtsRef(Board.GREENHOUSE, slug.lower() if slug else None, job_id) if slug else None
    if host == "jobs.lever.co" and segments:
        return AtsRef(
            Board.LEVER, segments[0].lower(), _uuid(segments[1] if len(segments) > 1 else None)
        )
    if host == "jobs.ashbyhq.com" and segments:
        return AtsRef(
            Board.ASHBY, segments[0].lower(), _uuid(segments[1] if len(segments) > 1 else None)
        )

    gh_jid = query.get("gh_jid", [""])[0]
    if gh_jid.isdigit():
        return AtsRef(Board.GREENHOUSE, None, gh_jid)
    return None
