"""Fetch job descriptions. Everything returned here is untrusted text and must never be logged."""

import html
import ipaddress
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from radar.ats import Board, parse_ats_url
from radar.net import NetworkError, request_with_retries

MIN_PAGE_TEXT_CHARS = 400
MAX_PAGE_BYTES = 5_000_000
MAX_REDIRECTS = 5
LEVER_MIN_INTERVAL = 1.05
LOCALE_SEGMENT = re.compile(r"[a-z]{2}-[A-Z]{2}")
SKIPPED_TAGS = frozenset({"script", "style", "noscript", "svg", "head", "template"})
BLOCK_TAGS = frozenset(
    {"p", "div", "br", "li", "ul", "ol", "tr", "section", "article", "header", "footer",
     "h1", "h2", "h3", "h4", "h5", "h6", "table", "blockquote"}
)  # fmt: skip


@dataclass(frozen=True)
class Description:
    text: str | None
    via: str

    @property
    def found(self) -> bool:
        return self.text is not None


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIPPED_TAGS:
            self._skip_depth += 1
        elif tag in BLOCK_TAGS:
            self._parts.append("\n- " if tag == "li" else "\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIPPED_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self._parts).splitlines())
        return "\n".join(line for line in lines if line and line != "-")


def html_to_text(markup: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(markup)
    extractor.close()
    return extractor.text()


_BOILERPLATE = re.compile(
    r"equal (?:employment )?opportunity|\beeo\b|affirmative action|without regard to"
    r"|protected (?:veteran|characteristic|class|status)|reasonable accommodation"
    r"|e-verify|fair chance|(?:arrest|conviction) records?|pay transparency"
    r"|privacy (?:policy|notice)|know your rights|unsolicited (?:resumes|applications)"
    r"|recruitment (?:agencies|fraud)",
    re.IGNORECASE,
)
_ELIGIBILITY = re.compile(
    r"relocat|housing|sponsor|security clearance|clearance (?:is )?required|itar"
    r"|export control|authorized to work|work authori[sz]ation|u\.?s\.? citizens? (?:only|required)"
    r"|must be a u\.?s\.? citizen|citizenship (?:is )?required",
    re.IGNORECASE,
)


def strip_boilerplate(text: str) -> str:
    """Drop legal boilerplate lines (EEO, accommodation, privacy and similar notices).

    These lines don't help score a posting but are often a quarter of a description. A
    line that mentions relocation, sponsorship, clearance or work authorization is always
    kept, since the scorer reports those.
    """
    kept = [
        line
        for line in text.splitlines()
        if not _BOILERPLATE.search(line) or _ELIGIBILITY.search(line)
    ]
    return "\n".join(kept)


def truncate_description(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[description truncated]"


def lever_text(posting: dict[str, Any]) -> str:
    sections = [posting.get("openingPlain"), posting.get("descriptionPlain")]
    for item in posting.get("lists") or []:
        sections.append(item.get("text"))
        sections.append(html_to_text(item.get("content") or ""))
    sections.append(posting.get("additionalPlain"))
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def is_safe_url(url: str) -> bool:
    """Only public https hosts. Feed URLs are third-party input, so a link (or a redirect)
    must not be able to point the runner at localhost, cloud metadata or private networks."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return False
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not host or port not in (None, 443):
        return False
    try:
        ipaddress.ip_address(host)
        return False
    except ValueError:
        pass
    return "." in host and not host.endswith((".local", ".internal", ".localhost", ".arpa"))


class DescriptionFetcher:
    def __init__(
        self,
        client: httpx.Client,
        max_chars: int,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._client = client
        self._max_chars = max_chars
        self._sleep = sleep or time.sleep
        self._ashby_boards: dict[str, dict[str, str]] = {}
        self._last_lever_request = 0.0

    def fetch(self, url: str) -> Description:
        if not is_safe_url(url):
            return Description(None, "unsafe_url")
        structured = self._structured_source(url)
        if structured is not None:
            via, fetch = structured
            text = self._guarded(fetch)
            if text:
                return self._finish(text, via)
        text = self._guarded(lambda: self._page(url))
        if text and len(text) >= MIN_PAGE_TEXT_CHARS:
            return self._finish(text, "page")
        return Description(None, "not_found")

    def _structured_source(self, url: str) -> tuple[str, Callable[[], str | None]] | None:
        ref = parse_ats_url(url)
        if ref is not None and ref.slug and ref.job_id:
            fetchers = {Board.GREENHOUSE: self._greenhouse, Board.LEVER: self._lever,
                        Board.ASHBY: self._ashby}  # fmt: skip
            return str(ref.board), partial(fetchers[ref.board], ref.slug, ref.job_id)
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if host.endswith(".myworkdayjobs.com"):
            segments = [segment for segment in parts.path.split("/") if segment]
            return "workday", partial(self._workday, host, segments)
        return None

    def _finish(self, text: str, via: str) -> Description:
        return Description(truncate_description(text, self._max_chars), via)

    @staticmethod
    def _guarded(fetch: Callable[[], str | None]) -> str | None:
        try:
            return fetch()
        except (
            NetworkError,
            httpx.HTTPError,
            httpx.InvalidURL,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
        ):
            return None

    def _get_json(self, url: str, label: str, **headers: str) -> Any | None:
        response = request_with_retries(
            self._client, "GET", url, label=label, headers=headers, sleep=self._sleep
        )
        if response.status_code != 200:
            return None
        return response.json()

    def _greenhouse(self, board: str, job: str) -> str | None:
        url = f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job}"
        payload = self._get_json(url, "greenhouse job")
        content = (payload or {}).get("content")
        return html_to_text(html.unescape(content)) if content else None

    def _lever(self, slug: str, posting_id: str) -> str | None:
        elapsed = time.monotonic() - self._last_lever_request
        if elapsed < LEVER_MIN_INTERVAL:
            self._sleep(LEVER_MIN_INTERVAL - elapsed)
        try:
            url = f"https://api.lever.co/v0/postings/{slug}/{posting_id}"
            payload = self._get_json(url, "lever job")
        finally:
            self._last_lever_request = time.monotonic()
        return lever_text(payload) if payload else None

    def _ashby(self, org: str, job_id: str) -> str | None:
        if org not in self._ashby_boards:
            url = f"https://api.ashbyhq.com/posting-api/job-board/{org}"
            payload = self._get_json(url, "ashby board") or {}
            self._ashby_boards[org] = {
                job["id"]: job.get("descriptionPlain")
                or html_to_text(job.get("descriptionHtml") or "")
                for job in payload.get("jobs") or []
                if job.get("id")
            }
        return self._ashby_boards[org].get(job_id) or None

    def _workday(self, host: str, segments: list[str]) -> str | None:
        """Workday's career pages load postings from this JSON endpoint; it is public but
        undocumented, so any failure falls back to the plain page fetch."""
        if segments and LOCALE_SEGMENT.fullmatch(segments[0]):
            segments = segments[1:]
        if len(segments) < 2:
            return None
        tenant, site, rest = host.split(".")[0], segments[0], "/".join(segments[1:])
        url = f"https://{host}/wday/cxs/{tenant}/{site}/{rest}"
        payload = self._get_json(url, "workday job", Accept="application/json")
        description = ((payload or {}).get("jobPostingInfo") or {}).get("jobDescription")
        return html_to_text(description) if description else None

    def _page(self, url: str) -> str | None:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            response = request_with_retries(
                self._client, "GET", current, label="job page",
                follow_redirects=False, sleep=self._sleep,
            )  # fmt: skip
            if response.is_redirect:
                location = response.headers.get("Location")
                current = urljoin(current, location) if location else ""
                if not is_safe_url(current):
                    return None
                continue
            content_type = response.headers.get("Content-Type", "")
            if (
                response.status_code != 200
                or "html" not in content_type
                or len(response.content) > MAX_PAGE_BYTES
            ):
                return None
            return html_to_text(response.text)
        return None
