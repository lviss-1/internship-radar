from typing import Any

import httpx2
import pytest
from pydantic import ValidationError

from radar.config import Settings
from radar.drafts import (
    DRAFT_INSTRUCTIONS,
    FALLBACK_BETA,
    Drafter,
    DraftResult,
    ResumeEdit,
    keep_grounded_edits,
)
from radar.scoring import ScoreResult
from radar.state import UsageLedger
from tests.conftest import SWE_RESUME, FakeClaude, anthropic_fixture, make_budget, make_posting

SCORE = ScoreResult(
    resume="swe",
    score=88,
    reasons=["Python backend"],
    gaps=["No Go"],
    relocation_mentioned=False,
    eligibility_flags=[],
)


def drafts_with(*originals: str) -> DraftResult:
    return DraftResult(
        resume_edits=[ResumeEdit(original=o, suggested="better", why="fit") for o in originals],
        cover_note="note",
        referral_message="hi",
    )


def test_keeps_edits_that_quote_the_resume() -> None:
    drafts = drafts_with("Built a REST API in Python for course scheduling")

    kept, dropped = keep_grounded_edits(drafts, SWE_RESUME)

    assert dropped == 0
    assert len(kept.resume_edits) == 1


@pytest.mark.parametrize(
    "original",
    [
        "• built a REST API in Python  for course scheduling",
        "- Built a REST API in Python for course scheduling",
        "BUILT A REST API IN PYTHON FOR COURSE SCHEDULING",
    ],
)
def test_matching_ignores_bullets_case_and_spacing(original: str) -> None:
    _, dropped = keep_grounded_edits(drafts_with(original), SWE_RESUME)

    assert dropped == 0


@pytest.mark.parametrize(
    ("original", "expected_dropped"),
    [
        ("Led a team of 40 engineers at Google", 1),
        ("Built a REST API in Rust for course scheduling", 1),
        ("Built a REST API", 0),
    ],
)
def test_drops_edits_that_invent_or_alter_facts(original: str, expected_dropped: int) -> None:
    grounded = "Automated weekly reporting with SQL and Python"

    kept, dropped = keep_grounded_edits(drafts_with(original, grounded), SWE_RESUME)

    assert dropped == expected_dropped
    assert grounded in [edit.original for edit in kept.resume_edits]


@pytest.mark.parametrize(
    "mutation",
    [
        {"cover_note": "x" * 1501},
        {"referral_message": "x" * 801},
        {"resume_edits": [{"original": "a", "suggested": "b", "why": "c"}] * 6},
        {"resume_edits": [{"original": "", "suggested": "b", "why": "c"}]},
        {"extra": 1},
    ],
)
def test_invalid_drafts_are_rejected(mutation: dict[str, Any]) -> None:
    valid = {"resume_edits": [], "cover_note": "n", "referral_message": "r"}

    with pytest.raises(ValidationError):
        DraftResult.model_validate({**valid, **mutation})


def test_drafter_uses_strong_model_with_fallback_and_filters_edits(settings: Settings) -> None:
    fake = FakeClaude(
        lambda body: httpx2.Response(200, json=anthropic_fixture("drafts_message.json"))
    )
    drafter = Drafter(fake.client, settings.models.strong, settings.scoring, make_budget(settings))

    drafts, dropped = drafter.draft(make_posting(), "desc", SCORE, SWE_RESUME)

    body = fake.requests[0]
    assert body["model"] == "claude-sonnet-5-5"
    assert body["fallbacks"] == "default"
    assert FALLBACK_BETA in fake.headers[0]["anthropic-beta"]
    assert body["output_config"]["effort"] == "medium"
    assert "<scoring_notes>" in body["messages"][0]["content"]
    assert dropped == 1
    assert [edit.original for edit in drafts.resume_edits] == [
        "Built a REST API in Python for course scheduling"
    ]


def test_draft_prompt_covers_co_ops_and_apprenticeships() -> None:
    assert "internship, co-op or apprenticeship" in DRAFT_INSTRUCTIONS


def test_drafter_records_spend_as_draft(settings: Settings) -> None:
    ledger = UsageLedger()
    fake = FakeClaude(
        lambda body: httpx2.Response(200, json=anthropic_fixture("drafts_message.json"))
    )
    budget = make_budget(settings, ledger)
    drafter = Drafter(fake.client, settings.models.strong, settings.scoring, budget)

    drafter.draft(make_posting(), "desc", SCORE, SWE_RESUME)

    assert list(ledger.month_by_kind) == ["draft"]
    assert ledger.month_by_kind["draft"].calls == 1
    assert ledger.month_by_kind["draft"].usd == pytest.approx(ledger.month_usd)
