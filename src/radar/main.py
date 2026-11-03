import argparse
import logging
import os
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import anthropic
import httpx

from radar.config import DEFAULT_SETTINGS_PATH, ConfigError, Settings, load_settings
from radar.delivery import Decision, Evaluation, decide
from radar.discord import (
    DiscordError,
    DiscordWebhook,
    drafts_payload,
    score_card_payload,
    text_payload,
)
from radar.drafts import Drafter
from radar.filters import LocationKind, Match, PostingFilter, RejectReason
from radar.net import NetworkError, make_client
from radar.scoring import (
    Budget,
    BudgetExhausted,
    ModelCallFailed,
    Resumes,
    Scorer,
    make_anthropic_client,
)
from radar.sources.descriptions import DescriptionFetcher
from radar.sources.simplify import SimplifyFeedError, load_postings
from radar.state import FileStateStore, GitStateStore, RadarState, StateError, StateStore

log = logging.getLogger("radar")

MATCHES_ENV = "DISCORD_WEBHOOK_MATCHES"
ERRORS_ENV = "DISCORD_WEBHOOK_ERRORS"
PING_USER_ENV = "DISCORD_PING_USER_ID"
CONSECUTIVE_FAILURE_LIMIT = 3
KNOWN_ERRORS = (ConfigError, SimplifyFeedError, StateError, DiscordError, NetworkError)


@dataclass
class RunSummary:
    fetched: int = 0
    invalid: int = 0
    matched: int = 0
    already_handled: int = 0
    deferred: int = 0
    descriptions_found: int = 0
    scored: int = 0
    below_threshold: int = 0
    drafted: int = 0
    dropped_edits: int = 0
    posted: int = 0
    failed: int = 0
    state_size: int = 0
    month_usd: float = 0.0
    budget_note: str | None = None
    rejected: Counter[RejectReason] = field(default_factory=Counter)
    unscored: Counter[str] = field(default_factory=Counter)
    draft_failures: Counter[str] = field(default_factory=Counter)

    def describe(self) -> str:
        def counts(counter: Counter[str] | Counter[RejectReason]) -> str:
            return ", ".join(f"{key}={value}" for key, value in sorted(counter.items()))

        return (
            f"fetched={self.fetched} invalid={self.invalid} matched={self.matched} "
            f"already_handled={self.already_handled} deferred={self.deferred} "
            f"descriptions={self.descriptions_found} scored={self.scored} "
            f"below_threshold={self.below_threshold} drafted={self.drafted} "
            f"dropped_edits={self.dropped_edits} posted={self.posted} failed={self.failed} "
            f"state_size={self.state_size} month_usd={self.month_usd:.4f} "
            f"unscored[{counts(self.unscored)}] draft_failures[{counts(self.draft_failures)}] "
            f"rejected[{counts(self.rejected)}]"
        )


@dataclass(frozen=True)
class Hooks:
    matches: DiscordWebhook
    errors: DiscordWebhook
    ping_user_id: str | None


@dataclass(frozen=True)
class AiConfig:
    client: anthropic.Anthropic
    resumes: Resumes


class Evaluator:
    """Fetches the description, scores the posting and drafts for high scores."""

    def __init__(
        self, settings: Settings, http: httpx.Client, ai: AiConfig, budget: Budget
    ) -> None:
        self._settings = settings
        self._resumes = ai.resumes
        self._fetcher = DescriptionFetcher(http, settings.scoring.max_description_chars)
        self._scorer = Scorer(
            ai.client, settings.models.cheap, settings.scoring, ai.resumes, budget
        )
        self._drafter = Drafter(ai.client, settings.models.strong, settings.scoring, budget)

    def evaluate(self, match: Match, summary: RunSummary) -> Evaluation:
        posting = match.posting
        description = self._fetcher.fetch(str(posting.url))
        summary.descriptions_found += description.found
        try:
            score = self._scorer.score(posting, description.text)
        except BudgetExhausted as exc:
            summary.budget_note = str(exc)
            summary.unscored["budget"] += 1
            return Evaluation(match, description.found, unscored_reason="budget cap")
        except ModelCallFailed as exc:
            summary.unscored[exc.reason] += 1
            return Evaluation(match, description.found, unscored_reason=exc.reason)
        summary.scored += 1

        relocation_blocked = (
            match.location_kind is LocationKind.OTHER_US and not score.relocation_mentioned
        )
        if score.score < self._settings.thresholds.ping or relocation_blocked:
            return Evaluation(match, description.found, score=score)
        try:
            drafts, dropped = self._drafter.draft(
                posting, description.text, score, self._resumes.get(score.resume)
            )
        except BudgetExhausted as exc:
            summary.budget_note = str(exc)
            summary.draft_failures["budget"] += 1
            return Evaluation(match, description.found, score=score)
        except ModelCallFailed as exc:
            summary.draft_failures[exc.reason] += 1
            return Evaluation(match, description.found, score=score)
        summary.drafted += 1
        summary.dropped_edits += dropped
        return Evaluation(match, description.found, score=score, drafts=drafts)


def describe_exception(exc: BaseException) -> str:
    """Our own exceptions carry vetted messages. Anything else could embed posting data or
    secrets in its text, so only the type and location are reported."""
    if isinstance(exc, KNOWN_ERRORS):
        return f"{type(exc).__name__}: {exc}"
    frames = traceback.extract_tb(exc.__traceback__)
    where = f" at {Path(frames[-1].filename).name}:{frames[-1].lineno}" if frames else ""
    return f"unexpected {type(exc).__name__}{where}"


def webhook_from_env(client: httpx.Client, env_name: str) -> DiscordWebhook:
    url = os.environ.get(env_name, "").strip()
    if not url:
        raise ConfigError(f"environment variable {env_name} is not set")
    return DiscordWebhook(client, url, label=env_name)


def ping_user_from_env() -> str | None:
    value = os.environ.get(PING_USER_ENV, "").strip()
    if not value:
        log.warning("%s is not set; high scores will post without a mention", PING_USER_ENV)
        return None
    if not value.isdigit():
        raise ConfigError(f"{PING_USER_ENV} must be a numeric Discord user ID")
    return value


def post_evaluations(
    to_post: list[tuple[Evaluation, Decision]],
    state: RadarState,
    store: StateStore,
    hooks: Hooks,
    summary: RunSummary,
) -> None:
    """Post score cards whose IDs were already reserved in state, then release failures.

    Like stamping an envelope "sent" before it goes in the mailbox: a crash can lose a
    post, but a re-run can never send it twice.
    """
    failed: list[str] = []
    consecutive_failures = 0
    for index, (evaluation, decision) in enumerate(to_post):
        key = evaluation.match.posting.key
        if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
            failed.extend(item.match.posting.key for item, _ in to_post[index:])
            log.error("stopping after %d consecutive Discord failures", consecutive_failures)
            break
        try:
            hooks.matches.send(score_card_payload(evaluation, decision, hooks.ping_user_id))
        except DiscordError as exc:
            failed.append(key)
            consecutive_failures += 1
            log.warning("post failed for %s: %s", key, exc)
            continue
        consecutive_failures = 0
        summary.posted += 1
        log.info("posted %s", key)
        if evaluation.drafts is not None:
            try:
                hooks.matches.send(drafts_payload(evaluation))
            except DiscordError as exc:
                summary.draft_failures["discord"] += 1
                log.warning("drafts post failed for %s: %s", key, exc)

    if failed:
        state.posted -= set(failed)
        store.persist(state, f"release {len(failed)} failed postings")
    summary.failed = len(failed)


def run(
    settings: Settings,
    client: httpx.Client,
    store: StateStore,
    hooks: Hooks | None,
    ai: AiConfig | None,
    now: datetime,
) -> RunSummary:
    summary = RunSummary()
    simplify = settings.sources.simplify
    feed = load_postings(client, simplify.url, simplify.max_invalid_ratio)
    summary.fetched = feed.total
    summary.invalid = feed.invalid

    posting_filter = PostingFilter(
        settings.filters,
        settings.locations,
        settings.excluded_companies,
        max_age=timedelta(hours=settings.delivery.max_post_age_hours),
    )
    result = posting_filter.apply(feed.postings, now)
    summary.matched = len(result.matches)
    summary.rejected = result.rejected

    state = store.load()
    state.usage.roll(local_date(now, settings))
    fresh = sorted(
        (match for match in result.matches if not state.handled(match.posting.key)),
        key=lambda match: match.posting.posted_at,
    )
    summary.already_handled = summary.matched - len(fresh)
    batch = fresh[: settings.delivery.max_posts_per_run]
    summary.deferred = len(fresh) - len(batch)

    if hooks is None or ai is None:
        for match in batch:
            log.info("dry run: would evaluate %s", match.posting.key)
        summary.state_size = len(state.posted)
        summary.month_usd = state.usage.month_usd
        return summary

    budget = Budget(
        state.usage,
        settings.budget,
        settings.pricing,
        settings.scoring.worst_case_input_tokens,
    )
    evaluator = Evaluator(settings, client, ai, budget)
    to_post: list[tuple[Evaluation, Decision]] = []
    for match in batch:
        evaluation = evaluator.evaluate(match, summary)
        decision = decide(evaluation, settings.thresholds, settings.quiet_hours, now)
        key = match.posting.key
        if decision.post:
            to_post.append((evaluation, decision))
        else:
            state.scored_below.add(key)
            summary.below_threshold += 1
            log.info("below threshold %s", key)

    alert_budget = bool(summary.budget_note) and state.usage.budget_alert_day != state.usage.day
    if alert_budget:
        state.usage.budget_alert_day = state.usage.day

    if batch:
        state.posted |= {evaluation.match.posting.key for evaluation, _ in to_post}
        store.persist(
            state,
            f"reserve {len(to_post)} postings, {summary.below_threshold} below threshold",
        )
        post_evaluations(to_post, state, store, hooks, summary)

    summary.state_size = len(state.posted)
    summary.month_usd = state.usage.month_usd
    if alert_budget:
        report_error(
            hooks.errors,
            f"{summary.budget_note}. New matches post unscored until it resets "
            f"(month to date ${state.usage.month_usd:.2f} of "
            f"${settings.budget.monthly_usd_hard_cap:.2f}).",
        )
    return summary


def local_date(now: datetime, settings: Settings) -> date:
    return now.astimezone(ZoneInfo(settings.quiet_hours.timezone)).date()


def problems_text(summary: RunSummary) -> str | None:
    unscored = {reason: n for reason, n in summary.unscored.items() if reason != "budget"}
    drafts = {reason: n for reason, n in summary.draft_failures.items() if reason != "budget"}
    parts = []
    if summary.failed:
        parts.append(f"{summary.failed} postings failed to send and will retry next run")
    if unscored:
        detail = ", ".join(f"{reason}={n}" for reason, n in sorted(unscored.items()))
        parts.append(f"{sum(unscored.values())} postings could not be scored ({detail}) "
                     "and were posted unscored")  # fmt: skip
    if drafts:
        detail = ", ".join(f"{reason}={n}" for reason, n in sorted(drafts.items()))
        parts.append(f"{sum(drafts.values())} drafts failed ({detail})")
    return "; ".join(parts) + "." if parts else None


def heartbeat_text(summary: RunSummary, now: datetime, settings: Settings) -> str:
    local = now.astimezone(ZoneInfo(settings.quiet_hours.timezone))
    return (
        f"Heartbeat {local:%Y-%m-%d %H:%M %Z}: radar is running. "
        f"This run fetched {summary.fetched} listings, {summary.matched} passed filters, "
        f"{summary.posted} new posted. {summary.state_size} postings delivered so far. "
        f"API spend this month: ${summary.month_usd:.2f} of "
        f"${settings.budget.monthly_usd_hard_cap:.2f}."
    )


def report_error(hook: DiscordWebhook | None, message: str) -> None:
    if hook is None:
        return
    try:
        hook.send(text_payload(f"Internship Radar error: {message}", silent=False))
    except DiscordError as exc:
        log.error("could not report to errors channel: %s", exc)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="radar", description="Find and post new internships.")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS_PATH)
    parser.add_argument("--state-dir", type=Path, default=Path(".state"))
    parser.add_argument(
        "--push-state",
        action="store_true",
        help="state-dir is a checkout of the state branch; commit and push after each write",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and filter only; never call Claude, post or write state",
    )
    parser.add_argument("--heartbeat", action="store_true", help="also post a daily heartbeat")
    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "httpx2", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    now = datetime.now(UTC)
    store: StateStore = (
        GitStateStore(args.state_dir) if args.push_state else FileStateStore(args.state_dir)
    )

    with make_client() as client:
        errors_hook: DiscordWebhook | None = None
        ai: AiConfig | None = None
        try:
            settings = load_settings(args.settings)
            hooks: Hooks | None = None
            if not args.dry_run:
                errors_hook = webhook_from_env(client, ERRORS_ENV)
                hooks = Hooks(
                    matches=webhook_from_env(client, MATCHES_ENV),
                    errors=errors_hook,
                    ping_user_id=ping_user_from_env(),
                )
                ai = AiConfig(make_anthropic_client(settings.scoring), Resumes.from_env())

            summary = run(settings, client, store, hooks, ai, now)
            log.info("run complete: %s", summary.describe())

            problems = problems_text(summary)
            if problems:
                report_error(errors_hook, problems)
            if args.heartbeat and errors_hook is not None:
                errors_hook.send(text_payload(heartbeat_text(summary, now, settings)))
            return 1 if summary.failed else 0
        except Exception as exc:
            message = describe_exception(exc)
            log.error("run failed: %s", message)
            report_error(errors_hook, message)
            return 1
        finally:
            if ai is not None:
                ai.client.close()


if __name__ == "__main__":
    sys.exit(main())
