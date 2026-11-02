import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import httpx

from radar.filters import LocationKind, Match
from radar.net import NetworkError, request_with_retries

SUPPRESS_NOTIFICATIONS = 1 << 12
TITLE_LIMIT = 256
FIELD_LIMIT = 1024
CONTENT_LIMIT = 2000
USERNAME = "Internship Radar"
WEBHOOK_HOSTS = frozenset(
    {"discord.com", "discordapp.com", "ptb.discord.com", "canary.discord.com"}
)

LOCATION_COLORS = {
    LocationKind.IN_PERSON: 0x2ECC71,
    LocationKind.REMOTE_US: 0x3498DB,
    LocationKind.OTHER_US: 0x95A5A6,
}
LOCATION_LABELS = {
    LocationKind.IN_PERSON: "In person (NYC/NJ/CT)",
    LocationKind.REMOTE_US: "Remote (US)",
    LocationKind.OTHER_US: "Other US (needs relocation)",
}


class DiscordError(Exception):
    pass


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _validate_webhook_url(url: str, label: str) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in WEBHOOK_HOSTS:
        raise DiscordError(f"{label}: webhook URL is not an https Discord URL")
    if not parts.path.startswith("/api/webhooks/"):
        raise DiscordError(f"{label}: webhook URL does not point at /api/webhooks/")


class DiscordWebhook:
    """Posts to one webhook. The URL is a secret, so it never appears in errors or repr."""

    def __init__(
        self,
        client: httpx.Client,
        url: str,
        label: str,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        _validate_webhook_url(url, label)
        self._client = client
        self._url = url
        self.label = label
        self._sleep = sleep or time.sleep

    def __repr__(self) -> str:
        return f"DiscordWebhook(label={self.label!r})"

    def send(self, payload: dict[str, Any]) -> None:
        try:
            response = request_with_retries(
                self._client,
                "POST",
                self._url,
                label=self.label,
                params={"wait": "true"},
                json=payload,
                attempts=4,
                sleep=self._sleep,
            )
        except NetworkError as exc:
            raise DiscordError(str(exc)) from exc
        if response.status_code not in (200, 204):
            raise DiscordError(f"{self.label}: HTTP {response.status_code}")
        self._respect_bucket(response)

    def _respect_bucket(self, response: httpx.Response) -> None:
        """Wait out the rate-limit bucket proactively instead of collecting 429s."""
        if response.headers.get("X-RateLimit-Remaining") != "0":
            return
        try:
            reset_after = float(response.headers.get("X-RateLimit-Reset-After", "0"))
        except ValueError:
            return
        if reset_after > 0:
            self._sleep(min(reset_after, 30.0))


def _field(name: str, value: str) -> dict[str, Any]:
    return {"name": name, "value": _truncate(value, FIELD_LIMIT), "inline": True}


def match_payload(match: Match) -> dict[str, Any]:
    posting = match.posting
    fields = [
        _field("Location", ", ".join(posting.locations) or "Unknown"),
        _field("Fit", LOCATION_LABELS[match.location_kind]),
        _field("Season", ", ".join(posting.terms) or "N/A"),
        _field("Posted", f"<t:{int(posting.posted_at.timestamp())}:R>"),
    ]
    if match.tags:
        fields.append(_field("Tags", ", ".join(match.tags)))

    footer = posting.source if not posting.category else f"{posting.source} · {posting.category}"
    embed = {
        "title": _truncate(f"{posting.company} — {posting.title}", TITLE_LIMIT),
        "url": str(posting.url),
        "color": LOCATION_COLORS[match.location_kind],
        "fields": fields,
        "footer": {"text": footer},
        "timestamp": posting.posted_at.isoformat(),
    }
    return {
        "username": USERNAME,
        "embeds": [embed],
        "allowed_mentions": {"parse": []},
        "flags": SUPPRESS_NOTIFICATIONS if match.silent else 0,
    }


def text_payload(text: str, *, silent: bool = True) -> dict[str, Any]:
    return {
        "username": USERNAME,
        "content": _truncate(text, CONTENT_LIMIT),
        "allowed_mentions": {"parse": []},
        "flags": SUPPRESS_NOTIFICATIONS if silent else 0,
    }
