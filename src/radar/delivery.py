from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from radar.config import QuietHours, Thresholds
from radar.drafts import DraftResult
from radar.filters import LocationKind, Match, Tag
from radar.scoring import ScoreResult

UNSCORED = "unscored"
NO_DESCRIPTION = "no description"
RELOCATION_MENTIONED = "relocation mentioned"
NO_RELOCATION = "no relocation mentioned"


@dataclass(frozen=True)
class Evaluation:
    match: Match
    description_found: bool
    score: ScoreResult | None = None
    drafts: DraftResult | None = None
    unscored_reason: str | None = None


@dataclass(frozen=True)
class Decision:
    post: bool
    silent: bool = True
    ping: bool = False
    tags: tuple[str, ...] = ()


def in_quiet_hours(now: datetime, quiet: QuietHours) -> bool:
    local = now.astimezone(ZoneInfo(quiet.timezone)).time()
    if quiet.start <= quiet.end:
        return quiet.start <= local < quiet.end
    return local >= quiet.start or local < quiet.end


def decide(
    evaluation: Evaluation, thresholds: Thresholds, quiet: QuietHours, now: datetime
) -> Decision:
    """Turn a scored match into what to do on Discord.

    Unscored matches still post, silently, so a scoring outage never hides a posting.
    Out-of-area roles stay silent unless relocation is mentioned, even at a high score.
    """
    match = evaluation.match
    tags = [tag for tag in match.tags if tag is not Tag.RELOCATION_UNKNOWN or not evaluation.score]
    if not evaluation.description_found:
        tags.append(NO_DESCRIPTION)

    score = evaluation.score
    if score is None:
        return Decision(post=True, silent=True, ping=False, tags=(UNSCORED, *tags))
    if score.score < thresholds.post:
        return Decision(post=False)

    needs_relocation = match.location_kind is LocationKind.OTHER_US
    if needs_relocation:
        tags.append(RELOCATION_MENTIONED if score.relocation_mentioned else NO_RELOCATION)
    silent = (needs_relocation and not score.relocation_mentioned) or in_quiet_hours(now, quiet)
    ping = score.score >= thresholds.ping and not silent
    return Decision(post=True, silent=silent, ping=ping, tags=tuple(tags))
