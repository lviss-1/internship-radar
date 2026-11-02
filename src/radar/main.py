import argparse
import logging
import os
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from radar.config import DEFAULT_SETTINGS_PATH, ConfigError, Settings, load_settings
from radar.discord import DiscordError, DiscordWebhook, match_payload, text_payload
from radar.filters import Match, PostingFilter, RejectReason
from radar.net import NetworkError, make_client
from radar.sources.simplify import SimplifyFeedError, load_postings
from radar.state import FileStateStore, GitStateStore, StateError, StateStore

log = logging.getLogger("radar")

MATCHES_ENV = "DISCORD_WEBHOOK_MATCHES"
ERRORS_ENV = "DISCORD_WEBHOOK_ERRORS"
CONSECUTIVE_FAILURE_LIMIT = 3
KNOWN_ERRORS = (ConfigError, SimplifyFeedError, StateError, DiscordError, NetworkError)


@dataclass
class RunSummary:
    fetched: int = 0
    invalid: int = 0
    matched: int = 0
    already_posted: int = 0
    posted: int = 0
    failed: int = 0
    deferred: int = 0
    state_size: int = 0
    rejected: Counter[RejectReason] = field(default_factory=Counter)

    def describe(self) -> str:
        rejected = ", ".join(f"{reason}={count}" for reason, count in sorted(self.rejected.items()))
        return (
            f"fetched={self.fetched} invalid={self.invalid} matched={self.matched} "
            f"already_posted={self.already_posted} posted={self.posted} failed={self.failed} "
            f"deferred={self.deferred} state_size={self.state_size} rejected[{rejected}]"
        )


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


def post_matches(
    to_post: list[Match], seen: set[str], store: StateStore, hook: DiscordWebhook
) -> tuple[int, list[str]]:
    """Reserve IDs in state before posting, then release any that failed.

    Like stamping an envelope "sent" before it goes in the mailbox: a crash can lose a
    post, but a re-run can never send it twice.
    """
    reserved = seen | {match.posting.key for match in to_post}
    store.persist(reserved, f"reserve {len(to_post)} postings")

    posted = 0
    failed: list[str] = []
    consecutive_failures = 0
    for index, match in enumerate(to_post):
        key = match.posting.key
        if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
            failed.extend(remaining.posting.key for remaining in to_post[index:])
            log.error("stopping after %d consecutive Discord failures", consecutive_failures)
            break
        try:
            hook.send(match_payload(match))
        except DiscordError as exc:
            failed.append(key)
            consecutive_failures += 1
            log.warning("post failed for %s: %s", key, exc)
            continue
        posted += 1
        consecutive_failures = 0
        log.info("posted %s", key)

    if failed:
        store.persist(reserved - set(failed), f"release {len(failed)} failed postings")
    return posted, failed


def run(
    settings: Settings,
    client: httpx.Client,
    store: StateStore,
    matches_hook: DiscordWebhook | None,
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

    seen = store.load()
    fresh = sorted(
        (match for match in result.matches if match.posting.key not in seen),
        key=lambda match: match.posting.posted_at,
    )
    summary.already_posted = summary.matched - len(fresh)
    to_post = fresh[: settings.delivery.max_posts_per_run]
    summary.deferred = len(fresh) - len(to_post)
    summary.state_size = len(seen)

    if matches_hook is None:
        for match in to_post:
            log.info("dry run: would post %s", match.posting.key)
        return summary

    if to_post:
        posted, failed = post_matches(to_post, seen, store, matches_hook)
        summary.posted = posted
        summary.failed = len(failed)
        summary.state_size = len(seen) + posted
    return summary


def heartbeat_text(summary: RunSummary, now: datetime) -> str:
    local = now.astimezone(ZoneInfo("America/New_York"))
    return (
        f"Heartbeat {local:%Y-%m-%d %H:%M %Z}: radar is running. "
        f"This run fetched {summary.fetched} listings, {summary.matched} passed filters, "
        f"{summary.posted} new posted. {summary.state_size} postings delivered so far."
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
        "--dry-run", action="store_true", help="fetch and filter only; never post or write state"
    )
    parser.add_argument("--heartbeat", action="store_true", help="also post a daily heartbeat")
    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore"):
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
        try:
            settings = load_settings(args.settings)
            if not args.dry_run:
                errors_hook = webhook_from_env(client, ERRORS_ENV)
            matches_hook = None if args.dry_run else webhook_from_env(client, MATCHES_ENV)

            summary = run(settings, client, store, matches_hook, now)
            log.info("run complete: %s", summary.describe())

            if summary.failed:
                report_error(errors_hook, f"{summary.failed} postings failed; retrying next run.")
            if args.heartbeat and errors_hook is not None:
                errors_hook.send(text_payload(heartbeat_text(summary, now), silent=True))
            return 1 if summary.failed else 0
        except Exception as exc:
            message = describe_exception(exc)
            log.error("run failed: %s", message)
            report_error(errors_hook, message)
            return 1


if __name__ == "__main__":
    sys.exit(main())
