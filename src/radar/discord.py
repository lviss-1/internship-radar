import re
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from radar.delivery import RELOCATION_MENTIONED, Decision, Evaluation
from radar.feedback import DRAFT_EMOJI, DRAFTS_PREFIX
from radar.filters import LocationKind, Match
from radar.net import NetworkError, request_with_retries
from radar.normalize import Posting
from radar.scoring import ScoreResult

SUPPRESS_NOTIFICATIONS = 1 << 12
TITLE_LIMIT = 256
FIELD_LIMIT = 1024
CONTENT_LIMIT = 2000
DESCRIPTION_LIMIT = 4096
EMBED_TOTAL_LIMIT = 5900
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
    LocationKind.IN_PERSON: "In person (NYC area/NJ/CT)",
    LocationKind.REMOTE_US: "Remote (US)",
    LocationKind.OTHER_US: "Other US (needs relocation)",
}
DRAFTS_FIELD = "Drafts"
DRAFTS_HINT = f"React {DRAFT_EMOJI} for resume edits, a cover note and a referral message."
FOOTER_SEPARATOR = " · "
REF_MARKER = " · ref "
_SCORE_HEADLINE = re.compile(r"^(\d{1,3})/100 · (SWE|PM) resume$")


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


def _field(name: str, value: str, *, inline: bool = True) -> dict[str, Any]:
    return {"name": name, "value": _truncate(value, FIELD_LIMIT), "inline": inline}


def _bullets(items: list[str]) -> str:
    return "\n".join(f"• {item}" for item in items)


def _embed_size(embed: dict[str, Any]) -> int:
    fields = embed.get("fields", [])
    return (
        len(embed.get("title", ""))
        + len(embed.get("description", ""))
        + len(embed.get("footer", {}).get("text", ""))
        + sum(len(f["name"]) + len(f["value"]) for f in fields)
    )


def _shrink_to_fit(embed: dict[str, Any]) -> dict[str, Any]:
    """Discord rejects a message whose embeds total more than 6000 characters, so trim the
    longest text slot until the embed fits."""
    slots = [(item, "value") for item in embed.get("fields", [])]
    if embed.get("description"):
        slots.append((embed, "description"))
    while slots and (overflow := _embed_size(embed) - EMBED_TOTAL_LIMIT) > 0:
        container, key = max(slots, key=lambda slot: len(slot[0][slot[1]]))
        current = container[key]
        trimmed = _truncate(current, max(20, len(current) - overflow))
        if trimmed == current:
            break
        container[key] = trimmed
    return embed


def _posting_embed(evaluation: Evaluation, title_prefix: str = "") -> dict[str, Any]:
    posting = evaluation.match.posting
    footer = posting.source if not posting.category else f"{posting.source} · {posting.category}"
    return {
        "title": _truncate(f"{title_prefix}{posting.company} — {posting.title}", TITLE_LIMIT),
        "url": str(posting.url),
        "color": LOCATION_COLORS[evaluation.match.location_kind],
        "footer": {"text": footer},
        "timestamp": posting.posted_at.isoformat(),
    }


def score_card_payload(
    evaluation: Evaluation, decision: Decision, ping_user_id: str | None
) -> dict[str, Any]:
    match = evaluation.match
    posting = match.posting
    score = evaluation.score
    headline = (
        f"{score.score}/100 · {score.resume.upper()} resume"
        if score
        else f"Unscored ({evaluation.unscored_reason or 'unknown'})"
    )
    fields = [
        _field("Score", headline),
        _field("Fit", LOCATION_LABELS[match.location_kind]),
        _field("Location", ", ".join(posting.locations) or "Unknown"),
        _field("Season", ", ".join(posting.terms) or "N/A"),
        _field("Posted", f"<t:{int(posting.posted_at.timestamp())}:R>"),
    ]
    if score and score.reasons:
        fields.append(_field("Why it fits", _bullets(score.reasons), inline=False))
    if score and score.gaps:
        fields.append(_field("Gaps", _bullets(score.gaps), inline=False))
    if score and score.eligibility_flags:
        fields.append(_field("Eligibility flags", _bullets(score.eligibility_flags), inline=False))
    if decision.tags:
        fields.append(_field("Tags", ", ".join(decision.tags), inline=False))
    if score and decision.draftable:
        fields.append(_field(DRAFTS_FIELD, DRAFTS_HINT, inline=False))

    embed = _posting_embed(evaluation)
    embed["footer"]["text"] += f"{REF_MARKER}{posting.key}"
    embed = _shrink_to_fit({**embed, "fields": fields})
    mention = ping_user_id if decision.ping and ping_user_id else None
    payload: dict[str, Any] = {
        "username": USERNAME,
        "embeds": [embed],
        "allowed_mentions": {"parse": [], "users": [mention] if mention else []},
        "flags": SUPPRESS_NOTIFICATIONS if decision.silent else 0,
    }
    if mention:
        payload["content"] = f"<@{mention}>"
    return payload


def drafts_payload(evaluation: Evaluation) -> dict[str, Any]:
    drafts = evaluation.drafts
    if drafts is None:
        raise ValueError("evaluation has no drafts")
    fields = [_field("Referral message", drafts.referral_message, inline=False)]
    for number, edit in enumerate(drafts.resume_edits, start=1):
        text = f"**Original:** {edit.original}\n**Suggested:** {edit.suggested}"
        if edit.why:
            text += f"\n**Why:** {edit.why}"
        fields.append(_field(f"Resume edit {number}", text, inline=False))
    if not drafts.resume_edits:
        fields.append(_field("Resume edits", "No grounded edits suggested.", inline=False))

    embed = {
        **_posting_embed(evaluation, title_prefix=DRAFTS_PREFIX),
        "description": _truncate(f"**Cover note**\n{drafts.cover_note}", DESCRIPTION_LIMIT),
        "fields": fields,
    }
    return {
        "username": USERNAME,
        "embeds": [_shrink_to_fit(embed)],
        "allowed_mentions": {"parse": []},
        "flags": SUPPRESS_NOTIFICATIONS,
    }


def _bullet_lines(value: str) -> list[str]:
    return [line.removeprefix("• ").strip() for line in value.splitlines() if line.strip()]


def card_request(embed: dict[str, Any]) -> Evaluation | None:
    """Rebuild the posting and score from a score card that offered drafts.

    The inverse of `score_card_payload`. The card is the only record of a posting's title
    and score, because the public state branch holds IDs only. Returns None for anything
    that isn't a draftable card in the current format, including cards posted before drafts
    became on-request.
    """
    fields = {item.get("name"): item.get("value") or "" for item in embed.get("fields") or []}
    title = embed.get("title") or ""
    footer = (embed.get("footer") or {}).get("text") or ""
    base, marker, key = footer.rpartition(REF_MARKER)
    company, dash, role = title.partition(" — ")
    headline = _SCORE_HEADLINE.match(fields.get("Score", ""))
    kind = next((k for k, label in LOCATION_LABELS.items() if label == fields.get("Fit")), None)
    if (
        DRAFTS_FIELD not in fields
        or title.startswith(DRAFTS_PREFIX)
        or not (marker and dash and headline and kind)
    ):
        return None

    source, _, source_id = key.partition(":")
    category = base.partition(FOOTER_SEPARATOR)[2] or None
    location = fields.get("Location", "")
    season = fields.get("Season", "")
    tags = fields.get("Tags", "").split(", ")
    try:
        posting = Posting(
            source=source,  # type: ignore[arg-type]
            source_id=source_id,
            company=company,
            title=role,
            locations=[location] if location and location != "Unknown" else [],
            url=embed.get("url") or "",
            posted_at=datetime.fromisoformat(embed.get("timestamp") or ""),
            terms=[season] if season and season != "N/A" else [],
            category=category,
        )
        score = ScoreResult(
            resume=headline.group(2).lower(),  # type: ignore[arg-type]
            score=int(headline.group(1)),
            reasons=_bullet_lines(fields.get("Why it fits", "")),
            gaps=_bullet_lines(fields.get("Gaps", "")),
            relocation_mentioned=RELOCATION_MENTIONED in tags,
            eligibility_flags=_bullet_lines(fields.get("Eligibility flags", "")),
        )
    except (ValidationError, ValueError):
        return None
    if posting.key != key or base.partition(FOOTER_SEPARATOR)[0] != source:
        return None
    return Evaluation(Match(posting, kind), description_found=False, score=score)


def text_payload(text: str, *, silent: bool = True) -> dict[str, Any]:
    return {
        "username": USERNAME,
        "content": _truncate(text, CONTENT_LIMIT),
        "allowed_mentions": {"parse": []},
        "flags": SUPPRESS_NOTIFICATIONS if silent else 0,
    }
