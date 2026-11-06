import importlib.util
import sys
from types import ModuleType
from typing import Any

import httpx2
import pytest

from radar.config import Settings
from radar.scoring import ScoreResult
from tests.conftest import RESUMES, ROOT, FakeClaude, make_posting, message_json


def load_eval() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_scoring", ROOT / "scripts/eval_scoring.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_scoring"] = module
    spec.loader.exec_module(module)
    return module


evaluation = load_eval()


def score(value: int, resume: str = "swe", relocation: bool = False) -> ScoreResult:
    return ScoreResult(
        resume=resume,  # type: ignore[arg-type]
        score=value,
        reasons=[],
        gaps=[],
        relocation_mentioned=relocation,
        eligibility_flags=[],
    )


def test_compare_counts_threshold_agreement_on_postings_both_scored() -> None:
    baseline = [score(85), score(65), score(55), None, score(90)]
    candidate = [
        score(78),
        score(70),
        score(61, resume="pm"),
        score(80),
        score(92, relocation=True),
    ]

    agreement = evaluation.compare(baseline, candidate, post=60, ping=80)

    assert agreement.compared == 4
    assert agreement.mean_abs_delta == pytest.approx((7 + 5 + 6 + 2) / 4)
    assert (agreement.post, agreement.ping) == (0.75, 0.75)
    assert (agreement.resume, agreement.relocation) == (0.75, 0.75)


def test_compare_with_nothing_in_common_is_empty() -> None:
    agreement = evaluation.compare([None, score(80)], [score(80), None], post=60, ping=80)

    assert agreement.compared == 0
    assert not evaluation.within_noise(agreement, agreement)


def test_within_noise_allows_a_small_margin_below_the_floor() -> None:
    floor = evaluation.Agreement(30, 3.0, 0.95, 0.90, 1.0, 1.0)
    close = evaluation.Agreement(30, 4.0, 0.91, 0.86, 1.0, 1.0)
    worse = evaluation.Agreement(30, 9.0, 0.95, 0.80, 1.0, 1.0)

    assert evaluation.within_noise(close, floor)
    assert not evaluation.within_noise(worse, floor)


def with_scoring(settings: Settings, **changes: object) -> Settings:
    return settings.model_copy(update={"scoring": settings.scoring.model_copy(update=changes)})


def test_variants_start_with_production_and_its_repeat(settings: Settings) -> None:
    unstripped = with_scoring(
        settings, strip_boilerplate=False, max_description_chars=15000, score_effort="medium"
    )

    plan = evaluation.variants(unstripped, 8000)

    assert [v.name for v in plan] == [
        "production", "repeat", "strip", "strip+8000", "strip+8000+low",
    ]  # fmt: skip
    production, repeat = plan[0], plan[1]
    assert (production.strip, production.max_chars) == (False, 15000)
    assert (production.strip, production.max_chars, production.effort) == (
        repeat.strip, repeat.max_chars, repeat.effort,
    )  # fmt: skip
    assert plan[-1].effort == "low"


def test_variants_skip_candidates_identical_to_production(settings: Settings) -> None:
    stripped = with_scoring(
        settings, strip_boilerplate=True, max_description_chars=8000, score_effort="medium"
    )

    plan = evaluation.variants(stripped, 8000)

    assert [v.name for v in plan] == ["production", "repeat", "strip+8000+low"]
    assert (plan[0].strip, plan[0].max_chars, plan[0].effort) == (True, 8000, "medium")


def test_variant_prepare_strips_then_truncates() -> None:
    variant = evaluation.Variant("strip+40", True, 40, "medium")
    text = "Build APIs in Python.\nWe are an equal opportunity employer.\n" + "x" * 100

    prepared = variant.prepare(text)

    assert "equal opportunity" not in prepared
    assert prepared.endswith("[description truncated]")


def test_run_variant_scores_each_posting_with_its_effort_and_own_ledger(
    settings: Settings,
) -> None:
    payload = {
        "resume": "swe",
        "score": 81,
        "reasons": [],
        "gaps": [],
        "relocation_mentioned": False,
        "eligibility_flags": [],
    }
    fake = FakeClaude(
        lambda body: httpx2.Response(200, json=message_json("claude-haiku-5-5", payload))
    )
    variant = evaluation.Variant("strip+8000+low", True, 8000, "low")
    sample: list[tuple[Any, str]] = [(make_posting(source_id=str(i)), "desc") for i in range(3)]

    run = evaluation.run_variant(variant, sample, fake.client, settings, RESUMES, cap_usd=0.5)

    assert [s.score for s in run.scores] == [81, 81, 81]
    assert {body["output_config"]["effort"] for body in fake.requests} == {"low"}
    assert run.ledger.month_by_kind["score"].calls == 3
    assert "usd=" in evaluation.describe(run, None, "")


def test_run_variant_stops_spending_at_its_cap(settings: Settings) -> None:
    fake = FakeClaude(lambda body: httpx2.Response(500))
    variant = evaluation.Variant("production", False, 15000, "medium")
    sample: list[tuple[Any, str]] = [(make_posting(source_id=str(i)), "desc") for i in range(3)]

    run = evaluation.run_variant(variant, sample, fake.client, settings, RESUMES, cap_usd=0.001)

    assert run.scores == [None, None, None]
    assert fake.requests == []
