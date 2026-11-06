"""Read reactions on past match posts: 👍/👎 so the scorer can calibrate to them, and ✍️ as a
request for drafts.

Nothing read here is stored: the private Discord channel stays the only record of what was
liked, and titles are never logged.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

import httpx

from radar.net import NetworkError, request_with_retries

log = logging.getLogger(__name__)

DISCORD_API = "https://discord.com/api/v10"
PAGE_SIZE = 100
MAX_PAGES = 3
MAX_AGE = timedelta(days=60)
MAX_PER_VERDICT = 10
THUMBS_UP = "\N{THUMBS UP SIGN}"
THUMBS_DOWN = "\N{THUMBS DOWN SIGN}"
DRAFT_EMOJI = "\N{WRITING HAND}"
DRAFTS_PREFIX = "Drafts: "
_SCORE_VALUE = re.compile(r"^(\d{1,3})/100")

Verdict = Literal["liked", "disliked"]


class FeedbackError(Exception):
    pass


@dataclass(frozen=True)
class FeedbackConfig:
    bot_token: str
    webhook_url: str

    def __repr__(self) -> str:
        return "FeedbackConfig(<redacted>)"


@dataclass(frozen=True)
class FeedbackExample:
    verdict: Verdict
    title: str
    score: int | None


@dataclass
class Feedback:
    liked: list[FeedbackExample] = field(default_factory=list)
    disliked: list[FeedbackExample] = field(default_factory=list)
    draft_cards: list[dict[str, Any]] = field(default_factory=list)
    messages_seen: int = 0
    with_embeds: int = 0

    @property
    def examples(self) -> list[FeedbackExample]:
        return [*self.liked, *self.disliked]


def _reacted(message: dict[str, Any], emoji: str) -> bool:
    """Skin tones and the emoji variation selector count as the same reaction."""
    return any(
        ((reaction.get("emoji") or {}).get("name") or "").startswith(emoji)
        and (reaction.get("count") or 0) > 0
        for reaction in message.get("reactions") or []
    )


def _verdict(message: dict[str, Any]) -> Verdict | None:
    """Exactly one of 👍/👎 counts; both or neither is no signal."""
    up, down = _reacted(message, THUMBS_UP), _reacted(message, THUMBS_DOWN)
    if up == down:
        return None
    return "liked" if up else "disliked"


def _score(embed: dict[str, Any]) -> int | None:
    for item in embed.get("fields") or []:
        if item.get("name") == "Score":
            match = _SCORE_VALUE.match(item.get("value") or "")
            return int(match.group(1)) if match else None
    return None


def _get(client: httpx.Client, url: str, label: str, **kwargs: Any) -> Any:
    try:
        response = request_with_retries(client, "GET", url, label=label, **kwargs)
    except NetworkError as exc:
        raise FeedbackError(str(exc)) from exc
    if response.status_code in (401, 403):
        raise FeedbackError(
            f"{label}: HTTP {response.status_code}; check the bot token and that the bot can "
            "view the matches channel and read its history"
        )
    if response.status_code != 200:
        raise FeedbackError(f"{label}: HTTP {response.status_code}")
    return response.json()


def read_feedback(client: httpx.Client, config: FeedbackConfig, now: datetime) -> Feedback:
    webhook = _get(client, config.webhook_url, "matches webhook")
    channel_id = str(webhook.get("channel_id") or "")
    if not channel_id.isdigit():
        raise FeedbackError("matches webhook: no channel ID in the webhook details")
    webhook_id = str(webhook.get("id") or "")

    feedback = Feedback()
    headers = {"Authorization": f"Bot {config.bot_token}"}
    before: str | None = None
    for _ in range(MAX_PAGES):
        params = {"limit": str(PAGE_SIZE), **({"before": before} if before else {})}
        url = f"{DISCORD_API}/channels/{channel_id}/messages"
        messages = _get(client, url, "discord bot", headers=headers, params=params)
        if not isinstance(messages, list):
            raise FeedbackError("discord bot: unexpected response")
        for message in messages:
            if now - datetime.fromisoformat(message["timestamp"]) > MAX_AGE:
                return _checked(feedback)
            _collect(message, feedback, webhook_id)
        if len(messages) < PAGE_SIZE:
            break
        before = messages[-1]["id"]
    return _checked(feedback)


def _collect(message: dict[str, Any], feedback: Feedback, webhook_id: str) -> None:
    """Draft requests count only on our own webhook's posts, since each one costs a model
    call and fetches the card's URL."""
    feedback.messages_seen += 1
    embeds = message.get("embeds") or []
    if not embeds:
        return
    feedback.with_embeds += 1
    title = embeds[0].get("title") or ""
    if not title or title.startswith(DRAFTS_PREFIX):
        return
    own_post = bool(webhook_id) and str(message.get("webhook_id") or "") == webhook_id
    if own_post and _reacted(message, DRAFT_EMOJI):
        feedback.draft_cards.append(embeds[0])
    verdict = _verdict(message)
    if verdict is None:
        return
    bucket = feedback.liked if verdict == "liked" else feedback.disliked
    if len(bucket) < MAX_PER_VERDICT:
        bucket.append(FeedbackExample(verdict, title, _score(embeds[0])))


def _checked(feedback: Feedback) -> Feedback:
    """Without the Message Content intent Discord returns posts with empty embeds, which
    would silently turn feedback off; say so instead. Messages arrive newest first, so draft
    requests are flipped to be handled oldest first."""
    if feedback.messages_seen and not feedback.with_embeds:
        raise FeedbackError(
            "message embeds came back empty; turn on the Message Content intent for the bot"
        )
    feedback.draft_cards.reverse()
    return feedback
