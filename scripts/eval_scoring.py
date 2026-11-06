"""Compare scoring variants on a live sample before changing scoring settings.

Scores the same matched postings under the production settings, under a repeat of them
(the noise floor: the model doesn't give identical scores twice), and under each
candidate. Then prints how often each candidate lands on the same side of the post and
ping thresholds as production. A candidate is safe when its agreement is about as good as
the repeat's.

Needs ANTHROPIC_API_KEY, RESUME_SWE_TEXT and RESUME_PM_TEXT, so it normally runs in the
eval-scoring workflow, where those are secrets. Output is counts and numbers only: never
titles, descriptions, resume text or model output, because Actions logs are public.

Spend here is capped by --budget-usd and is not added to the radar's own ledger.
"""

import argparse
import hashlib
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import anthropic

from radar.config import DEFAULT_SETTINGS_PATH, ConfigError, Effort, Settings, load_settings
from radar.filters import PostingFilter
from radar.net import make_client
from radar.normalize import Posting
from radar.scoring import (
    Budget,
    BudgetExhausted,
    ModelCallFailed,
    Resumes,
    Scorer,
    ScoreResult,
    make_anthropic_client,
)
from radar.sources.descriptions import DescriptionFetcher, strip_boilerplate, truncate_description
from radar.sources.simplify import SimplifyFeedError, load_postings
from radar.state import UsageLedger

log = logging.getLogger("eval_scoring")

NOISE_MARGIN = 0.05
FETCH_ATTEMPTS_PER_SAMPLE = 4


@dataclass(frozen=True)
class Variant:
    name: str
    strip: bool
    max_chars: int
    effort: Effort

    def prepare(self, description: str) -> str:
        text = strip_boilerplate(description) if self.strip else description
        return truncate_description(text, self.max_chars)


@dataclass(frozen=True)
class Agreement:
    compared: int
    mean_abs_delta: float
    post: float
    ping: float
    resume: float
    relocation: float


@dataclass(frozen=True)
class VariantRun:
    variant: Variant
    scores: list[ScoreResult | None]
    ledger: UsageLedger
    avg_chars: int


def variants(settings: Settings, candidate_chars: int) -> list[Variant]:
    """Production, its repeat, then each candidate that differs from production."""
    scoring = settings.scoring
    production = Variant(
        "production", scoring.strip_boilerplate, scoring.max_description_chars, scoring.score_effort
    )
    trimmed = min(candidate_chars, scoring.max_description_chars)
    candidates = [
        Variant("strip", True, production.max_chars, production.effort),
        Variant(f"strip+{trimmed}", True, trimmed, production.effort),
        Variant(f"strip+{trimmed}+low", True, trimmed, "low"),
    ]
    tried = {(production.strip, production.max_chars, production.effort)}
    result = [
        production,
        Variant("repeat", production.strip, production.max_chars, production.effort),
    ]
    for candidate in candidates:
        setting = (candidate.strip, candidate.max_chars, candidate.effort)
        if setting not in tried:
            tried.add(setting)
            result.append(candidate)
    return result


def compare(
    baseline: Sequence[ScoreResult | None],
    candidate: Sequence[ScoreResult | None],
    post: int,
    ping: int,
) -> Agreement:
    """Agreement on postings both runs scored. Each share is the fraction of those postings
    where the two runs made the same call."""
    pairs = [(a, b) for a, b in zip(baseline, candidate, strict=True) if a and b]
    if not pairs:
        return Agreement(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def share(same: list[bool]) -> float:
        return sum(same) / len(same)

    return Agreement(
        compared=len(pairs),
        mean_abs_delta=sum(abs(a.score - b.score) for a, b in pairs) / len(pairs),
        post=share([(a.score >= post) == (b.score >= post) for a, b in pairs]),
        ping=share([(a.score >= ping) == (b.score >= ping) for a, b in pairs]),
        resume=share([a.resume == b.resume for a, b in pairs]),
        relocation=share([a.relocation_mentioned == b.relocation_mentioned for a, b in pairs]),
    )


def within_noise(candidate: Agreement, floor: Agreement) -> bool:
    return (
        candidate.compared > 0
        and candidate.post >= floor.post - NOISE_MARGIN
        and candidate.ping >= floor.ping - NOISE_MARGIN
    )


def stable_order(postings: list[Posting]) -> list[Posting]:
    """A fixed shuffle, so re-running the eval picks mostly the same postings."""
    return sorted(postings, key=lambda p: hashlib.sha256(p.key.encode()).hexdigest())


def collect_sample(
    settings: Settings, size: int, max_age: timedelta, now: datetime
) -> list[tuple[Posting, str]]:
    """Matched postings that have a description; trimming only matters for those."""
    simplify = settings.sources.simplify
    with make_client() as http:
        feed = load_postings(http, simplify.url, simplify.max_invalid_ratio)
        posting_filter = PostingFilter(
            settings.filters, settings.locations, settings.excluded_companies, max_age=max_age
        )
        matches = posting_filter.apply(feed.postings, now).matches
        fetcher = DescriptionFetcher(http, settings.scoring.max_description_chars)
        sample: list[tuple[Posting, str]] = []
        candidates = stable_order([match.posting for match in matches])
        for posting in candidates[: size * FETCH_ATTEMPTS_PER_SAMPLE]:
            text = posting.description or fetcher.fetch(str(posting.url)).text
            if text:
                sample.append((posting, text))
            if len(sample) == size:
                break
    log.info("matched=%d sampled=%d", len(matches), len(sample))
    return sample


def run_variant(
    variant: Variant,
    sample: list[tuple[Posting, str]],
    client: anthropic.Anthropic,
    settings: Settings,
    resumes: Resumes,
    cap_usd: float,
) -> VariantRun:
    ledger = UsageLedger()
    budget_settings = settings.budget.model_copy(
        update={"monthly_usd_hard_cap": cap_usd, "daily_api_call_cap": len(sample) + 1}
    )
    budget = Budget(
        ledger, budget_settings, settings.pricing, settings.scoring.worst_case_input_tokens
    )
    scoring = settings.scoring.model_copy(update={"score_effort": variant.effort})
    scorer = Scorer(client, settings.models.cheap, scoring, resumes, budget)
    scores: list[ScoreResult | None] = []
    chars = 0
    for posting, description in sample:
        text = variant.prepare(description)
        chars += len(text)
        try:
            scores.append(scorer.score(posting, text))
        except (ModelCallFailed, BudgetExhausted):
            scores.append(None)
    return VariantRun(variant, scores, ledger, chars // max(1, len(sample)))


def describe(run: VariantRun, agreement: Agreement | None, verdict: str) -> str:
    totals = run.ledger.month_by_kind.get("score")
    metered = totals.metered_calls if totals else 0
    prompt = (
        (totals.input_tokens + totals.cache_read_tokens + totals.cache_write_tokens) // metered
        if totals and metered
        else 0
    )
    output = totals.output_tokens // metered if totals and metered else 0
    failed = sum(score is None for score in run.scores)
    line = (
        f"{run.variant.name:<22} effort={run.variant.effort:<6} avg_chars={run.avg_chars:<6} "
        f"avg_in={prompt:<6} avg_out={output:<5} usd={run.ledger.month_usd:.4f} failed={failed}"
    )
    if agreement is not None:
        line += (
            f" | n={agreement.compared} mean_abs_delta={agreement.mean_abs_delta:.1f} "
            f"agree@post={agreement.post:.0%} agree@ping={agreement.ping:.0%} "
            f"resume={agreement.resume:.0%} relocation={agreement.relocation:.0%} {verdict}"
        )
    return line


def between[N: (int, float)](kind: Callable[[str], N], low: N, high: N) -> Callable[[str], N]:
    def parse(value: str) -> N:
        number = kind(value)
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
        return number

    return parse


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare scoring variants on live postings.")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS_PATH)
    parser.add_argument("--sample", type=between(int, 5, 100), default=30)
    parser.add_argument("--chars", type=between(int, 1000, 50000), default=8000)
    parser.add_argument("--max-age-days", type=between(int, 1, 60), default=14)
    parser.add_argument("--budget-usd", type=between(float, 0.05, 5.0), default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore", "httpx2", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    now = datetime.now(UTC)
    try:
        settings = load_settings(args.settings)
        resumes = Resumes.from_env()
        client = make_anthropic_client(settings.scoring)
    except ConfigError as exc:
        log.error("setup failed: %s", exc)
        return 1

    try:
        sample = collect_sample(settings, args.sample, timedelta(days=args.max_age_days), now)
    except SimplifyFeedError as exc:
        log.error("feed failed: %s", exc)
        return 1
    if len(sample) < 5:
        log.error("only %d postings with descriptions; too few to compare", len(sample))
        return 1

    plan = variants(settings, args.chars)
    cap = args.budget_usd / len(plan)
    with client:
        runs = [run_variant(v, sample, client, settings, resumes, cap) for v in plan]

    post, ping = settings.thresholds.post, settings.thresholds.ping
    production, *others = runs
    floor = compare(production.scores, others[0].scores, post, ping)
    log.info(describe(production, None, ""))
    log.info(describe(others[0], floor, "(noise floor)"))
    for run in others[1:]:
        agreement = compare(production.scores, run.scores, post, ping)
        verdict = "OK" if within_noise(agreement, floor) else "WORSE THAN NOISE"
        log.info(describe(run, agreement, verdict))
    total = sum(run.ledger.month_usd for run in runs)
    log.info("total eval spend: $%.4f (cap $%.2f)", total, args.budget_usd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
