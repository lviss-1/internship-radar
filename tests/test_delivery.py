from datetime import UTC, datetime

import pytest

from radar.config import Settings
from radar.delivery import (
    NO_DESCRIPTION,
    NO_RELOCATION,
    RELOCATION_MENTIONED,
    UNSCORED,
    Evaluation,
    decide,
    in_quiet_hours,
)
from radar.filters import LocationKind, Match, Tag
from radar.scoring import ScoreResult
from tests.conftest import NOW, make_posting


def scored(value: int, relocation: bool = False) -> ScoreResult:
    return ScoreResult(
        resume="swe",
        score=value,
        reasons=[],
        gaps=[],
        relocation_mentioned=relocation,
        eligibility_flags=[],
    )


def evaluation(
    kind: LocationKind = LocationKind.IN_PERSON,
    score: ScoreResult | None = None,
    description_found: bool = True,
    tags: tuple[Tag, ...] = (),
) -> Evaluation:
    match = Match(make_posting(), kind, tags)
    return Evaluation(match, description_found, score=score, unscored_reason=None)


def eastern(hour: int, minute: int = 0) -> datetime:
    """Build an ET wall-clock time on 2026-11-02 (EST, UTC-5)."""
    return datetime(2026, 11, 2, hour + 5, minute, tzinfo=UTC)


@pytest.mark.parametrize(
    ("hour", "minute", "quiet"),
    [(0, 59, False), (1, 0, True), (3, 30, True), (6, 59, True), (7, 0, False), (12, 0, False)],
)
def test_quiet_hours_boundaries(settings: Settings, hour: int, minute: int, quiet: bool) -> None:
    assert in_quiet_hours(eastern(hour, minute), settings.quiet_hours) is quiet


def test_quiet_hours_follow_daylight_saving(settings: Settings) -> None:
    summer_1am_edt = datetime(2026, 7, 1, 5, 0, tzinfo=UTC)

    assert in_quiet_hours(summer_1am_edt, settings.quiet_hours)


def test_below_post_threshold_is_not_posted(settings: Settings) -> None:
    decision = decide(evaluation(score=scored(59)), settings.thresholds, settings.quiet_hours, NOW)

    assert not decision.post


def test_mid_score_posts_with_notification_but_no_ping(settings: Settings) -> None:
    decision = decide(evaluation(score=scored(60)), settings.thresholds, settings.quiet_hours, NOW)

    assert (decision.post, decision.silent, decision.ping) == (True, False, False)


def test_high_score_pings(settings: Settings) -> None:
    decision = decide(evaluation(score=scored(80)), settings.thresholds, settings.quiet_hours, NOW)

    assert (decision.post, decision.silent, decision.ping) == (True, False, True)


def test_high_score_during_quiet_hours_is_silent(settings: Settings) -> None:
    decision = decide(
        evaluation(score=scored(95)), settings.thresholds, settings.quiet_hours, eastern(2)
    )

    assert (decision.post, decision.silent, decision.ping) == (True, True, False)


def test_out_of_area_without_relocation_stays_silent_even_when_high(settings: Settings) -> None:
    item = evaluation(LocationKind.OTHER_US, scored(95), tags=(Tag.RELOCATION_UNKNOWN,))

    decision = decide(item, settings.thresholds, settings.quiet_hours, NOW)

    assert (decision.post, decision.silent, decision.ping) == (True, True, False)
    assert decision.tags == (NO_RELOCATION,)


def test_out_of_area_with_relocation_pings(settings: Settings) -> None:
    item = evaluation(LocationKind.OTHER_US, scored(85, relocation=True))

    decision = decide(item, settings.thresholds, settings.quiet_hours, NOW)

    assert (decision.silent, decision.ping) == (False, True)
    assert RELOCATION_MENTIONED in decision.tags


def test_unscored_posts_silently_and_keeps_filter_tags(settings: Settings) -> None:
    item = evaluation(
        LocationKind.OTHER_US,
        score=None,
        description_found=False,
        tags=(Tag.SEASON_UNCONFIRMED, Tag.RELOCATION_UNKNOWN),
    )

    decision = decide(item, settings.thresholds, settings.quiet_hours, NOW)

    assert (decision.post, decision.silent, decision.ping) == (True, True, False)
    assert decision.tags == (
        UNSCORED,
        Tag.SEASON_UNCONFIRMED,
        Tag.RELOCATION_UNKNOWN,
        NO_DESCRIPTION,
    )


def test_remote_role_is_not_gated_on_relocation(settings: Settings) -> None:
    item = evaluation(LocationKind.REMOTE_US, scored(90))

    decision = decide(item, settings.thresholds, settings.quiet_hours, NOW)

    assert decision.ping
    assert RELOCATION_MENTIONED not in decision.tags and NO_RELOCATION not in decision.tags
