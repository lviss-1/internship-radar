"""Decide which errors-channel alerts to send, so a lasting problem doesn't post every run."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum


class AlertKind(StrEnum):
    CONDITION = "condition"
    """A state that can persist (feed down, board down). Repeats after the cooldown and gets
    a "recovered" line once it clears."""
    EVENT = "event"
    """Something that happened this run (a post failed). Repeats after the cooldown."""
    ONCE = "once"
    """Sent a single time; the category carries its own period, e.g. budget_target:2026-11."""


@dataclass(frozen=True)
class Alert:
    category: str
    kind: AlertKind
    message: str


RECOVERY_TEXT = {
    "simplify_failed": "Simplify feed is reachable again.",
    "boards_failed": "Board polling is healthy again.",
    "run_failed": "Runs are succeeding again.",
}


def _recovery_text(category: str) -> str:
    prefix, _, detail = category.partition(":")
    if prefix == "board_down":
        return f"Board {detail} is responding again."
    return RECOVERY_TEXT.get(prefix, f"{category} cleared.")


CONDITION_PREFIXES = frozenset({"simplify_failed", "boards_failed", "board_down", "run_failed"})


def _is_condition(category: str) -> bool:
    return category.split(":", 1)[0] in CONDITION_PREFIXES


def plan_alerts(
    alerts: list[Alert],
    sent: dict[str, str],
    now: datetime,
    cooldown: timedelta,
    month: str,
) -> str | None:
    """Return the message to post (or None) and update `sent` in place.

    `sent` maps category to when it was last posted; it lives in the usage ledger on the
    state branch, so the cooldown holds across runs. Run this only after a run whose
    outcome is complete, since a missing condition is read as "recovered". `month` is the
    ledger's own (Eastern) month, so once-a-month alerts roll over with the budget.
    """
    lines: list[str] = []
    current = {alert.category for alert in alerts}

    for alert in alerts:
        last = sent.get(alert.category)
        if alert.kind is AlertKind.ONCE:
            due = last is None
        else:
            due = last is None or now - datetime.fromisoformat(last) >= cooldown
        if due:
            lines.append(alert.message)
            sent[alert.category] = now.isoformat()

    for category in sorted(set(sent) - current):
        if _is_condition(category):
            lines.append(f"Recovered: {_recovery_text(category)}")
            del sent[category]
        elif not category.startswith("budget_target:") and (
            now - datetime.fromisoformat(sent[category]) >= cooldown
        ):
            del sent[category]

    for category in [c for c in sent if c.startswith("budget_target:") and not c.endswith(month)]:
        del sent[category]

    return "\n".join(f"• {line}" for line in lines) if lines else None
