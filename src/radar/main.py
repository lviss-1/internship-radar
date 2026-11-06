import argparse
import logging
import os
import statistics
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import anthropic
import httpx

from radar.alerts import Alert, AlertKind, plan_alerts
from radar.config import DEFAULT_SETTINGS_PATH, ConfigError, Settings, load_settings
from radar.delivery import Decision, Evaluation, decide
from radar.discord import (
    DiscordError,
    DiscordWebhook,
    card_request,
    drafts_payload,
    score_card_payload,
    text_payload,
)
from radar.drafts import Drafter
from radar.feedback import Feedback, FeedbackConfig, FeedbackError, read_feedback
from radar.filters import Match, PostingFilter, RejectReason
from radar.net import NetworkError, make_client, request_with_retries
from radar.normalize import Posting, Source
from radar.scoring import (
    Budget,
    BudgetExhausted,
    ModelCallFailed,
    Resumes,
    Scorer,
    make_anthropic_client,
)
from radar.sources.boards import (
    BoardResult,
    BoardSource,
    WatchlistBoard,
    WatchlistError,
    load_watchlist,
)
from radar.sources.descriptions import Description, DescriptionFetcher, strip_boilerplate
from radar.sources.simplify import SimplifyFeedError, load_postings
from radar.state import (
    FileStateStore,
    GitStateStore,
    LagStats,
    RadarState,
    StateError,
    StateStore,
    UsageLedger,
    serialize_state,
)

log = logging.getLogger("radar")

MATCHES_ENV = "DISCORD_WEBHOOK_MATCHES"
ERRORS_ENV = "DISCORD_WEBHOOK_ERRORS"
PING_USER_ENV = "DISCORD_PING_USER_ID"
HEALTHCHECK_ENV = "HEALTHCHECK_PING_URL"
BOT_ENV = "DISCORD_BOT_TOKEN"
CONSECUTIVE_FAILURE_LIMIT = 3
DEFAULT_ALERT_COOLDOWN = timedelta(hours=6)
BOARD_FAILURE_ALERT_RATIO = 0.25
TRANSIENT_MODEL_FAILURES = frozenset({"rate_limited", "api_unreachable", "api_error"})
KNOWN_ERRORS = (
    ConfigError,
    SimplifyFeedError,
    StateError,
    DiscordError,
    NetworkError,
    WatchlistError,
)


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
    draft_requests: int = 0
    drafts_deferred: int = 0
    drafted: int = 0
    dropped_edits: int = 0
    posted: int = 0
    failed: int = 0
    state_size: int = 0
    month_usd: float = 0.0
    spend_text: str = ""
    budget_note: str | None = None
    heartbeat_due: bool = False
    simplify_failure: str | None = None
    boards_total: int = 0
    boards_polled: int = 0
    board_postings: int = 0
    board_failures: dict[str, str] = field(default_factory=dict)
    boards_down: dict[str, int] = field(default_factory=dict)
    watchlist_age_days: float | None = None
    feedback_liked: int = 0
    feedback_disliked: int = 0
    feedback_error: str | None = None
    lag_text: str = ""
    rejected: Counter[RejectReason] = field(default_factory=Counter)
    unscored: Counter[str] = field(default_factory=Counter)
    draft_failures: Counter[str] = field(default_factory=Counter)

    def describe(self) -> str:
        def counts(counter: Counter[str] | Counter[RejectReason]) -> str:
            return ", ".join(f"{key}={value}" for key, value in sorted(counter.items()))

        return (
            f"fetched={self.fetched} invalid={self.invalid} "
            f"boards={self.boards_polled}/{self.boards_total} board_postings={self.board_postings} "
            f"matched={self.matched} "
            f"already_handled={self.already_handled} deferred={self.deferred} "
            f"descriptions={self.descriptions_found} scored={self.scored} "
            f"below_threshold={self.below_threshold} draft_requests={self.draft_requests} "
            f"drafts_deferred={self.drafts_deferred} drafted={self.drafted} "
            f"dropped_edits={self.dropped_edits} posted={self.posted} failed={self.failed} "
            f"state_size={self.state_size} month_usd={self.month_usd:.4f} "
            f"spend[{self.spend_text}] "
            f"unscored[{counts(self.unscored)}] draft_failures[{counts(self.draft_failures)}] "
            f"rejected[{counts(self.rejected)}] lag[{self.lag_text}] "
            f"feedback liked={self.feedback_liked} disliked={self.feedback_disliked}"
        )

    @property
    def boards_alarming(self) -> bool:
        return bool(self.boards_total) and (
            len(self.board_failures) / self.boards_total >= BOARD_FAILURE_ALERT_RATIO
        )

    @property
    def source_failed(self) -> bool:
        return self.simplify_failure is not None or self.boards_alarming


@dataclass(frozen=True)
class Hooks:
    matches: DiscordWebhook
    errors: DiscordWebhook
    ping_user_id: str | None
    feedback: FeedbackConfig | None = None


@dataclass(frozen=True)
class AiConfig:
    client: anthropic.Anthropic
    resumes: Resumes


class Evaluator:
    """Fetches descriptions, scores new postings and drafts for cards someone asked about."""

    def __init__(
        self,
        settings: Settings,
        http: httpx.Client,
        ai: AiConfig,
        budget: Budget,
        feedback: Feedback | None = None,
    ) -> None:
        self._settings = settings
        self._resumes = ai.resumes
        self._fetcher = DescriptionFetcher(http, settings.scoring.max_description_chars)
        self._scorer = Scorer(
            ai.client, settings.models.cheap, settings.scoring, ai.resumes, budget, feedback
        )
        self._drafter = Drafter(ai.client, settings.models.strong, settings.scoring, budget)

    def _model_input(self, description: str | None) -> str | None:
        if description and self._settings.scoring.strip_boilerplate:
            return strip_boilerplate(description)
        return description

    def evaluate(self, match: Match, summary: RunSummary) -> Evaluation:
        posting = match.posting
        description = (
            Description(posting.description, posting.source)
            if posting.description
            else self._fetcher.fetch(str(posting.url))
        )
        summary.descriptions_found += description.found
        try:
            score = self._scorer.score(posting, self._model_input(description.text))
        except BudgetExhausted as exc:
            summary.budget_note = str(exc)
            summary.unscored["budget"] += 1
            return Evaluation(match, description.found, unscored_reason="budget cap")
        except ModelCallFailed as exc:
            summary.unscored[exc.reason] += 1
            return Evaluation(match, description.found, unscored_reason=exc.reason)
        summary.scored += 1
        return Evaluation(match, description.found, score=score)

    def draft(self, request: Evaluation) -> tuple[Evaluation, int]:
        """Draft for a card rebuilt by `card_request`. Raises BudgetExhausted and
        ModelCallFailed for the caller to handle."""
        if request.score is None:
            raise ValueError("draft requests always carry a score")
        posting = request.match.posting
        description = self._fetcher.fetch(str(posting.url))
        drafts, dropped = self._drafter.draft(
            posting,
            self._model_input(description.text),
            request.score,
            self._resumes.get(request.score.resume),
        )
        return replace(request, description_found=description.found, drafts=drafts), dropped


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


def feedback_config_from_env() -> FeedbackConfig | None:
    token = os.environ.get(BOT_ENV, "").strip()
    if not token:
        log.warning("%s is not set; scoring runs without reaction feedback", BOT_ENV)
        return None
    webhook_url = os.environ.get(MATCHES_ENV, "").strip()
    if not webhook_url:
        log.warning("%s is not set; reaction feedback needs it to find the channel", MATCHES_ENV)
        return None
    return FeedbackConfig(bot_token=token, webhook_url=webhook_url)


def check_feedback(client: httpx.Client, now: datetime) -> int:
    """Read reactions once and log only counts, to confirm the bot setup works."""
    config = feedback_config_from_env()
    if config is None:
        return 1
    try:
        feedback = read_feedback(client, config, now)
    except FeedbackError as exc:
        log.error("feedback check failed: %s", exc)
        return 1
    log.info(
        "feedback check ok: messages=%d with_embeds=%d liked=%d disliked=%d draft_reactions=%d",
        feedback.messages_seen,
        feedback.with_embeds,
        len(feedback.liked),
        len(feedback.disliked),
        len(feedback.draft_cards),
    )
    return 0


def load_feedback(
    client: httpx.Client, config: FeedbackConfig | None, now: datetime, summary: RunSummary
) -> Feedback | None:
    """Read reactions for this run's scorer. A failure only costs the calibration examples."""
    if config is None:
        return None
    try:
        feedback = read_feedback(client, config, now)
    except FeedbackError as exc:
        summary.feedback_error = str(exc)
        log.warning("feedback unavailable: %s", exc)
        return None
    summary.feedback_liked, summary.feedback_disliked = len(feedback.liked), len(feedback.disliked)
    return feedback


def ping_user_from_env() -> str | None:
    value = os.environ.get(PING_USER_ENV, "").strip()
    if not value:
        log.warning("%s is not set; high scores will post without a mention", PING_USER_ENV)
        return None
    if not value.isdigit():
        raise ConfigError(f"{PING_USER_ENV} must be a numeric Discord user ID")
    return value


class StateWriter:
    """Saves state through the store, skipping saves when nothing changed since the last one,
    so quiet runs make no commits on the state branch."""

    def __init__(self, store: StateStore, state: RadarState) -> None:
        self.state = state
        self._store = store
        self._saved = serialize_state(state)

    def save(self, message: str) -> None:
        self._store.persist(self.state, message)
        self._saved = serialize_state(self.state)

    def save_if_changed(self, message: str) -> None:
        if serialize_state(self.state) != self._saved:
            self.save(message)


def post_evaluations(
    to_post: list[tuple[Evaluation, Decision]],
    writer: StateWriter,
    hooks: Hooks,
    summary: RunSummary,
) -> None:
    """Post score cards whose IDs were already reserved in state, then release failures.

    Like stamping an envelope "sent" before it goes in the mailbox: a crash can lose a
    post, but a re-run can never send it twice.
    """
    failed: list[Posting] = []
    consecutive_failures = 0
    for index, (evaluation, decision) in enumerate(to_post):
        key = evaluation.match.posting.key
        if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
            failed.extend(item.match.posting for item, _ in to_post[index:])
            log.error("stopping after %d consecutive Discord failures", consecutive_failures)
            break
        try:
            hooks.matches.send(score_card_payload(evaluation, decision, hooks.ping_user_id))
        except DiscordError as exc:
            failed.append(evaluation.match.posting)
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
        for posting in failed:
            writer.state.posted -= posting.keys
        writer.save(f"release {len(failed)} failed postings")
    summary.failed = len(failed)


def is_transient(reason: str) -> bool:
    return reason in TRANSIENT_MODEL_FAILURES or reason.startswith("api_http_5")


def pending_draft_requests(cards: list[dict[str, Any]], state: RadarState) -> list[Evaluation]:
    """Rebuild each reacted card once, skipping cards that never offered drafts and postings
    already drafted."""
    requests: list[Evaluation] = []
    keys: set[str] = set()
    for card in cards:
        request = card_request(card)
        if request is None:
            continue
        key = request.match.posting.key
        if key not in state.drafted and key not in keys:
            keys.add(key)
            requests.append(request)
    return requests


def fulfil_draft_requests(
    requests: list[Evaluation],
    evaluator: Evaluator,
    writer: StateWriter,
    hooks: Hooks,
    summary: RunSummary,
    limit: int,
) -> None:
    """Draft for cards someone reacted to, oldest first, at most `limit` per run.

    The key is reserved in state before the drafts post, so a re-run never posts them twice.
    A failure that would only repeat (a refusal, invalid output) is reserved too, so one bad
    card can't be charged for again every half hour; transient failures retry next run.
    """
    summary.draft_requests = len(requests)
    summary.drafts_deferred = max(0, len(requests) - limit)
    for request in requests[:limit]:
        key = request.match.posting.key
        try:
            evaluation, dropped = evaluator.draft(request)
        except BudgetExhausted as exc:
            summary.budget_note = str(exc)
            summary.draft_failures["budget"] += 1
            return
        except ModelCallFailed as exc:
            summary.draft_failures[exc.reason] += 1
            log.warning("drafting failed for %s: %s", key, exc.reason)
            if not is_transient(exc.reason):
                writer.state.drafted.add(key)
                writer.save("give up drafting 1 posting")
            continue

        writer.state.drafted.add(key)
        writer.save("reserve drafts for 1 posting")
        try:
            hooks.matches.send(drafts_payload(evaluation))
        except DiscordError as exc:
            writer.state.drafted.discard(key)
            writer.save("release 1 failed draft post")
            summary.draft_failures["discord"] += 1
            log.warning("drafts post failed for %s: %s", key, exc)
            continue
        summary.drafted += 1
        summary.dropped_edits += dropped
        log.info("drafted %s", key)


def merge_sources(
    board_postings: list[Posting],
    simplify_postings: list[Posting],
    published: dict[str, datetime],
    lag: LagStats,
) -> list[Posting]:
    """Combine sources, keeping the board copy when Simplify lists the same job.

    The board copy has the employer's real publish time and the description. Each overlap
    is also a free measurement of how long Simplify takes to list a job after the
    employer publishes it.
    """
    board_keys = set().union(*(posting.keys for posting in board_postings))
    merged = list(board_postings)
    for posting in simplify_postings:
        for alias in posting.keys - {posting.key}:
            if alias in published:
                delay = (posting.posted_at - published[alias]).total_seconds() / 60
                lag.record_simplify_delay(posting.key, delay)
        if not posting.keys & board_keys:
            merged.append(posting)
    return merged


def record_radar_lag(posting: Posting, lag: LagStats, now: datetime) -> None:
    """Count only postings published after their source was first polled, so the backlog
    found on a board's first poll doesn't read as hours of lag."""
    first_polled = lag.first_polled.get(posting.board or posting.source)
    if first_polled and posting.posted_at >= datetime.fromisoformat(first_polled):
        lag.record_radar(posting.source, (now - posting.posted_at).total_seconds() / 60)


def lag_report(lag: LagStats) -> str:
    def stats(samples: list[float]) -> str:
        if not samples:
            return "n=0"
        ordered = sorted(samples)
        p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
        return f"median {statistics.median(ordered):.0f}m p90 {p90:.0f}m n={len(ordered)}"

    parts = [f"{source}: {stats(samples)}" for source, samples in sorted(lag.radar_minutes.items())]
    parts.append(f"simplify after board: {stats(lag.simplify_delay_minutes)}")
    return "; ".join(parts)


def run(
    settings: Settings,
    client: httpx.Client,
    store: StateStore,
    hooks: Hooks | None,
    ai: AiConfig | None,
    now: datetime,
    force_heartbeat: bool = False,
    boards: list[WatchlistBoard] | None = None,
    watchlist_generated: datetime | None = None,
) -> RunSummary:
    summary = RunSummary()
    if watchlist_generated is not None:
        summary.watchlist_age_days = (now - watchlist_generated).total_seconds() / 86400
    state = store.load()
    writer = StateWriter(store, state)
    state.usage.roll(local_date(now, settings))
    posting_filter = PostingFilter(
        settings.filters,
        settings.locations,
        settings.excluded_companies,
        max_age=timedelta(hours=settings.delivery.max_post_age_hours),
    )

    simplify_postings: list[Posting] = []
    simplify = settings.sources.simplify
    try:
        feed = load_postings(client, simplify.url, simplify.max_invalid_ratio)
    except SimplifyFeedError as exc:
        summary.simplify_failure = str(exc)
        log.error("simplify failed: %s", exc)
    else:
        summary.fetched, summary.invalid = feed.total, feed.invalid
        simplify_postings = feed.postings
        state.lag.first_polled.setdefault("simplify", now.isoformat())
        backfilled = state.backfill_aliases((p.key, p.keys) for p in simplify_postings)
        if backfilled:
            log.info("backfilled %d board aliases for already-handled postings", backfilled)

    def wants(source: Source, title: str, posted_at: datetime, keys: frozenset[str]) -> bool:
        return posting_filter.worth_fetching(source, title, posted_at, now) and not state.handled(
            keys
        )

    board_result = BoardResult()
    if boards:
        board_source = BoardSource(client, settings.scoring.max_description_chars)
        board_result = board_source.fetch(boards, wants)
    summary.boards_total = len(boards or [])
    summary.boards_polled = len(board_result.polled)
    summary.board_postings = len(board_result.postings)
    summary.board_failures = board_result.failures
    for board_key in board_result.polled:
        state.lag.first_polled.setdefault(board_key, now.isoformat())
    for board_key, reason in sorted(board_result.failures.items()):
        log.warning("board %s failed: %s", board_key, reason)
    summary.boards_down = track_board_health(
        state.lag.board_failures, boards or [], board_result, settings.delivery.board_down_after
    )

    postings = merge_sources(
        board_result.postings, simplify_postings, board_result.published, state.lag
    )
    result = posting_filter.apply(postings, now)
    summary.matched = len(result.matches)
    summary.rejected = result.rejected

    fresh = sorted(
        (match for match in result.matches if not state.handled(match.posting.keys)),
        key=lambda match: match.posting.posted_at,
    )
    summary.already_handled = summary.matched - len(fresh)
    batch = fresh[: settings.delivery.max_posts_per_run]
    summary.deferred = len(fresh) - len(batch)
    summary.lag_text = lag_report(state.lag)

    if hooks is None or ai is None:
        for match in batch:
            log.info("dry run: would evaluate %s", match.posting.key)
        summary.state_size = len(state.posted)
        summary.month_usd = state.usage.month_usd
        summary.spend_text = spend_report(state.usage)
        return summary

    budget = Budget(
        state.usage,
        settings.budget,
        settings.pricing,
        settings.scoring.worst_case_input_tokens,
    )
    feedback = load_feedback(client, hooks.feedback, now, summary)
    evaluator = Evaluator(settings, client, ai, budget, feedback)
    to_post: list[tuple[Evaluation, Decision]] = []
    for match in batch:
        record_radar_lag(match.posting, state.lag, now)
        evaluation = evaluator.evaluate(match, summary)
        decision = decide(evaluation, settings.thresholds, settings.quiet_hours, now)
        if hooks.feedback is None:
            decision = replace(decision, draftable=False)
        if decision.post:
            to_post.append((evaluation, decision))
        else:
            state.scored_below |= match.posting.keys
            summary.below_threshold += 1
            log.info("below threshold %s", match.posting.key)
    summary.lag_text = lag_report(state.lag)

    alert_budget = bool(summary.budget_note) and state.usage.budget_alert_day != state.usage.day
    if alert_budget:
        state.usage.budget_alert_day = state.usage.day
    summary.heartbeat_due = force_heartbeat or heartbeat_is_due(state, now, settings)
    if summary.heartbeat_due:
        state.usage.heartbeat_day = state.usage.day

    if batch:
        for evaluation, _ in to_post:
            state.posted |= evaluation.match.posting.keys
        writer.save(f"reserve {len(to_post)} postings, {summary.below_threshold} below threshold")
        post_evaluations(to_post, writer, hooks, summary)

    if feedback is not None:
        fulfil_draft_requests(
            pending_draft_requests(feedback.draft_cards, state),
            evaluator,
            writer,
            hooks,
            summary,
            settings.delivery.max_drafts_per_run,
        )

    summary.state_size = len(state.posted)
    summary.month_usd = state.usage.month_usd
    summary.spend_text = spend_report(state.usage)
    alerts = collect_alerts(summary, state, settings)
    if alert_budget:
        alerts.append(
            Alert(
                f"budget_cap:{state.usage.day}",
                AlertKind.ONCE,
                f"{summary.budget_note}. New matches post unscored until it resets "
                f"(month to date ${state.usage.month_usd:.2f} of "
                f"${settings.budget.monthly_usd_hard_cap:.2f}).",
            )
        )
    send_alerts(alerts, state, hooks.errors, now, settings)
    writer.save_if_changed("record alerts, heartbeat and lag")
    return summary


def spend_report(usage: UsageLedger) -> str:
    """Month-to-date spend per call kind with average tokens per call. Counts only."""
    parts = []
    for kind, totals in sorted(usage.month_by_kind.items()):
        part = f"{kind} ${totals.usd:.2f} over {totals.calls} calls"
        if totals.metered_calls:
            prompt = totals.input_tokens + totals.cache_read_tokens + totals.cache_write_tokens
            cached = round(100 * totals.cache_read_tokens / prompt) if prompt else 0
            part += (
                f", avg {prompt // totals.metered_calls} in ({cached}% cache read)"
                f" / {totals.output_tokens // totals.metered_calls} out tokens"
            )
        parts.append(part)
    untracked = usage.month_usd - sum(totals.usd for totals in usage.month_by_kind.values())
    if untracked >= 0.005:
        parts.append(f"${untracked:.2f} from before per-kind tracking")
    return "; ".join(parts) or "no model calls yet"


def local_date(now: datetime, settings: Settings) -> date:
    return now.astimezone(ZoneInfo(settings.quiet_hours.timezone)).date()


def heartbeat_is_due(state: RadarState, now: datetime, settings: Settings) -> bool:
    """Once a day, on the first run after the configured hour, whatever triggered the run."""
    local = now.astimezone(ZoneInfo(settings.quiet_hours.timezone))
    return (
        local.hour >= settings.delivery.heartbeat_hour
        and state.usage.heartbeat_day != local.date().isoformat()
    )


def ping_healthcheck(client: httpx.Client, succeeded: bool) -> None:
    """Tell the external dead-man's switch how this run went. Never fails the run."""
    url = os.environ.get(HEALTHCHECK_ENV, "").strip()
    if not url:
        log.warning("%s is not set; skipping the healthcheck ping", HEALTHCHECK_ENV)
        return
    if not url.startswith("https://"):
        log.warning("%s is not an https URL; skipping the healthcheck ping", HEALTHCHECK_ENV)
        return
    target = url if succeeded else f"{url.rstrip('/')}/fail"
    try:
        response = request_with_retries(client, "POST", target, label="healthcheck", attempts=2)
    except NetworkError as exc:
        log.warning("healthcheck ping failed: %s", exc)
        return
    if response.status_code >= 400:
        log.warning("healthcheck ping returned HTTP %d", response.status_code)


def track_board_health(
    failures: dict[str, int],
    boards: list[WatchlistBoard],
    result: BoardResult,
    down_after: int,
) -> dict[str, int]:
    """Count consecutive failed polls per board and return the boards now considered down.

    One failure is usually a blip; several in a row usually means the company moved to a
    different job system, which only re-running the probe script fixes.
    """
    watched = {board.key for board in boards}
    for key in [key for key in failures if key not in watched]:
        del failures[key]
    for key in result.polled:
        failures.pop(key, None)
    for key in result.failures:
        failures[key] = failures.get(key, 0) + 1
    return {key: count for key, count in sorted(failures.items()) if count >= down_after}


def _counts(counter: Counter[str]) -> str:
    return ", ".join(f"{reason}={n}" for reason, n in sorted(counter.items()))


def collect_alerts(summary: RunSummary, state: RadarState, settings: Settings) -> list[Alert]:
    def condition(category: str, message: str) -> None:
        alerts.append(Alert(category, AlertKind.CONDITION, message))

    def event(category: str, message: str) -> None:
        alerts.append(Alert(category, AlertKind.EVENT, message))

    alerts: list[Alert] = []
    if summary.simplify_failure:
        condition("simplify_failed", f"Simplify feed failed ({summary.simplify_failure}).")
    if summary.boards_alarming:
        detail = ", ".join(f"{key}: {why}" for key, why in sorted(summary.board_failures.items()))
        failed = f"{len(summary.board_failures)} of {summary.boards_total}"
        condition("boards_failed", f"{failed} boards failed this run ({detail}).")
    for board_key, count in summary.boards_down.items():
        why = summary.board_failures.get(board_key, "unknown")
        condition(
            f"board_down:{board_key}",
            f"Board {board_key} has failed {count} runs in a row ({why}). The company may "
            "have changed job systems; re-run scripts/probe_boards.py.",
        )
    if summary.feedback_error:
        event(
            "feedback_unavailable", f"Reaction feedback is off this run ({summary.feedback_error})."
        )
    if summary.failed:
        event("post_failed", f"{summary.failed} postings failed to send; they retry next run.")
    unscored = Counter({why: n for why, n in summary.unscored.items() if why != "budget"})
    if unscored:
        event(
            "unscored",
            f"{unscored.total()} postings could not be scored ({_counts(unscored)}) "
            "and were posted unscored.",
        )
    drafts = Counter({why: n for why, n in summary.draft_failures.items() if why != "budget"})
    if drafts:
        event("drafts_failed", f"{drafts.total()} drafts failed ({_counts(drafts)}).")

    max_age = settings.delivery.watchlist_max_age_days
    if summary.watchlist_age_days is not None and summary.watchlist_age_days > max_age:
        alerts.append(
            Alert(
                f"watchlist_stale:{state.usage.month}",
                AlertKind.ONCE,
                f"The watchlist is {summary.watchlist_age_days:.0f} days old. Re-run "
                "`uv run python scripts/probe_boards.py --discover` and commit "
                "config/watchlist.yaml to pick up companies that started hiring.",
            )
        )

    usage, budget = state.usage, settings.budget
    if usage.month_usd >= budget.monthly_usd_target:
        alerts.append(
            Alert(
                f"budget_target:{usage.month}",
                AlertKind.ONCE,
                f"API spend reached ${usage.month_usd:.2f} this month, past the "
                f"${budget.monthly_usd_target:.2f} target "
                f"(hard cap ${budget.monthly_usd_hard_cap:.2f}).",
            )
        )
    return alerts


def send_alerts(
    alerts: list[Alert],
    state: RadarState,
    hook: DiscordWebhook,
    now: datetime,
    settings: Settings,
) -> None:
    """Post what's due and record it, but only record if the post went through, so a
    Discord hiccup doesn't silence an alert for the whole cooldown."""
    before = dict(state.usage.alerts_sent)
    cooldown = timedelta(hours=settings.delivery.alert_cooldown_hours)
    message = plan_alerts(alerts, state.usage.alerts_sent, now, cooldown, state.usage.month)
    if message and not report_error(hook, message):
        state.usage.alerts_sent = before


def alert_run_failure(
    store: StateStore,
    hook: DiscordWebhook | None,
    message: str,
    now: datetime,
    cooldown: timedelta,
) -> None:
    """Report a crashed run at most once per cooldown. If state itself can't be read or
    saved, report every time; the healthcheck /fail ping is the backstop either way."""
    try:
        state = store.load()
    except Exception:
        report_error(hook, f"Run failed: {message}")
        return
    last = state.usage.alerts_sent.get("run_failed")
    if last is not None and now - datetime.fromisoformat(last) < cooldown:
        log.info("run failure already reported within the cooldown")
        return
    if not report_error(hook, f"Run failed: {message}"):
        return
    state.usage.alerts_sent["run_failed"] = now.isoformat()
    try:
        store.persist(state, "record alert")
    except Exception:
        log.warning("could not record the run-failure alert")


def heartbeat_text(summary: RunSummary, now: datetime, settings: Settings) -> str:
    local = now.astimezone(ZoneInfo(settings.quiet_hours.timezone))
    return (
        f"Heartbeat {local:%Y-%m-%d %H:%M %Z}: radar is running. "
        f"This run fetched {summary.fetched} listings, {summary.matched} passed filters, "
        f"{summary.posted} new posted. {summary.state_size} postings delivered so far. "
        f"API spend this month: ${summary.month_usd:.2f} of "
        f"${settings.budget.monthly_usd_hard_cap:.2f} ({summary.spend_text}). "
        f"Boards polled: {summary.boards_polled} of {summary.boards_total}. "
        f"{down_boards_text(summary)}"
        f"Detection lag: {summary.lag_text}."
    )


def down_boards_text(summary: RunSummary) -> str:
    if not summary.boards_down:
        return ""
    return f"Boards down: {', '.join(summary.boards_down)}. "


def report_error(hook: DiscordWebhook | None, message: str) -> bool:
    """Post to the errors channel. Returns whether the post went through."""
    if hook is None:
        return False
    try:
        hook.send(text_payload(f"**Internship Radar alert**\n{message}", silent=False))
    except DiscordError as exc:
        log.error("could not report to errors channel: %s", exc)
        return False
    return True


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
    parser.add_argument(
        "--heartbeat", action="store_true", help="post the heartbeat even if one was sent today"
    )
    parser.add_argument(
        "--check-feedback",
        action="store_true",
        help="read reactions with the Discord bot, log counts only, and exit",
    )
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
        if args.check_feedback:
            return check_feedback(client, now)
        code = execute(args, client, store, now)
        if not args.dry_run:
            ping_healthcheck(client, succeeded=code == 0)
        return code


def execute(
    args: argparse.Namespace, client: httpx.Client, store: StateStore, now: datetime
) -> int:
    errors_hook: DiscordWebhook | None = None
    ai: AiConfig | None = None
    cooldown = DEFAULT_ALERT_COOLDOWN
    try:
        settings = load_settings(args.settings)
        cooldown = timedelta(hours=settings.delivery.alert_cooldown_hours)
        hooks: Hooks | None = None
        if not args.dry_run:
            errors_hook = webhook_from_env(client, ERRORS_ENV)
            hooks = Hooks(
                matches=webhook_from_env(client, MATCHES_ENV),
                errors=errors_hook,
                ping_user_id=ping_user_from_env(),
                feedback=feedback_config_from_env(),
            )
            ai = AiConfig(make_anthropic_client(settings.scoring), Resumes.from_env())

        watchlist = load_watchlist(settings.watchlist_path, settings.excluded_companies)
        summary = run(
            settings,
            client,
            store,
            hooks,
            ai,
            now,
            force_heartbeat=args.heartbeat,
            boards=watchlist.boards,
            watchlist_generated=watchlist.generated_at,
        )
        log.info("run complete: %s", summary.describe())

        if summary.heartbeat_due and errors_hook is not None:
            try:
                errors_hook.send(text_payload(heartbeat_text(summary, now, settings)))
            except DiscordError as exc:
                log.warning("heartbeat failed: %s", exc)
        return 1 if summary.failed or summary.source_failed else 0
    except Exception as exc:
        message = describe_exception(exc)
        log.error("run failed: %s", message)
        if errors_hook is not None:
            alert_run_failure(store, errors_hook, message, now, cooldown)
        return 1
    finally:
        if ai is not None:
            ai.client.close()


if __name__ == "__main__":
    sys.exit(main())
