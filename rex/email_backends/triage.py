"""Email triage categorization.

Assigns one of four triage categories to each :class:`EmailEnvelope`:

- ``urgent``          — time-sensitive, needs immediate attention
- ``action_required`` — requires a response or explicit action
- ``newsletter``      — bulk / digest / marketing content
- ``fyi``             — informational catch-all

Categorization inspects the sender address, subject line, and snippet
(body preview) in that priority order.  The first matching rule wins:
urgent → action_required → newsletter → fyi.
"""

from __future__ import annotations

import re

from rex.email_backends.base import EmailEnvelope

# ── Valid categories ─────────────────────────────────────────────────────────

TRIAGE_CATEGORIES: tuple[str, ...] = (
    "urgent",
    "action_required",
    "newsletter",
    "fyi",
)

# ── Pattern tables ───────────────────────────────────────────────────────────

_URGENT_SUBJECT_PATTERNS: list[str] = [
    r"\burgent\b",
    r"\bcritical\b",
    r"\bemergency\b",
    r"\boutage\b",
    r"\bincident\b",
    r"\balert\b",
    r"\bimmediate(ly)?\b",
    r"\bdown\b",
]

_URGENT_SENDER_PATTERNS: list[str] = [
    r"\balerts?@",
    r"\bmonitor(ing)?@",
    r"@.*monitor\.",
    r"pagerduty",
    r"opsgenie",
    r"cloudwatch",
    r"statuspage",
]

_URGENT_BODY_PATTERNS: list[str] = [
    r"\bjoin the incident bridge\b",
    r"\bimmediate action required\b",
    r"\bproduction.*down\b",
    r"\bsev[- ]?[12]\b",
]

_ACTION_REQUIRED_SUBJECT_PATTERNS: list[str] = [
    r"\baction\s+required\b",
    r"\bplease\s+complete\b",
    r"\bplease\s+review\b",
    r"\bplease\s+respond\b",
    r"\bdue\s+in\b",
    r"\bdue\s+by\b",
    r"\bdue\s+on\b",
    r"\bdue\s+this\b",
    r"\binvoice\b",
    r"\bpayment\b",
    r"\bannual\s+review\b",
    r"\bself[- ]review\b",
    r"\bsign\b.{0,20}\bnow\b",
    r"\bconfirm\b.{0,20}\bby\b",
    r"\brespond\b.{0,20}\bby\b",
    r"\bdeadline\b",
    r"\bexpires?\b",
]

_NEWSLETTER_SENDER_PATTERNS: list[str] = [
    r"\bnewsletter@",
    r"\bdigest@",
    r"\bupdates@",
    r"@.*newsletter",
    r"@.*digest",
    r"\bmarketing@",
    r"\bpromotion@",
    r"\bnewsletter\.",
]

_NEWSLETTER_SUBJECT_PATTERNS: list[str] = [
    r"\bnewsletter\b",
    r"\bweekly\b.{0,30}\b(digest|roundup|edition|update)\b",
    r"\bmonthly\b.{0,30}\b(digest|roundup|edition|update)\b",
    r"\bthis\s+week\s+in\b",
    r"\b(week|month)ly\s+update\b",
    r"\bedition\b",
    r"\bdigest\b",
    r"\bunsubscribe\b",
]


# ── Internal helpers ─────────────────────────────────────────────────────────


def _match(text: str, patterns: list[str]) -> bool:
    """Return True if *text* matches any pattern (case-insensitive)."""
    lower = text.lower()
    return any(re.search(pat, lower) for pat in patterns)


# ── Public API ───────────────────────────────────────────────────────────────


def categorize(envelope: EmailEnvelope) -> str:
    """Return the triage category for *envelope*.

    Inspection order: sender address → subject keywords → body / snippet.
    Category priority: urgent → action_required → newsletter → fyi.
    """
    subject = envelope.subject
    sender = envelope.from_addr
    body = envelope.snippet

    # 1. Urgent
    if (
        _match(subject, _URGENT_SUBJECT_PATTERNS)
        or _match(sender, _URGENT_SENDER_PATTERNS)
        or _match(body, _URGENT_BODY_PATTERNS)
    ):
        return "urgent"

    # 2. Action required
    if _match(subject, _ACTION_REQUIRED_SUBJECT_PATTERNS):
        return "action_required"

    # 3. Newsletter
    if _match(sender, _NEWSLETTER_SENDER_PATTERNS) or _match(
        subject, _NEWSLETTER_SUBJECT_PATTERNS
    ):
        return "newsletter"

    # 4. FYI — catch-all
    return "fyi"


def triage_emails(
    envelopes: list[EmailEnvelope],
) -> dict[str, list[EmailEnvelope]]:
    """Categorize *envelopes* and return them grouped by category.

    The returned dict always contains all four category keys.
    """
    result: dict[str, list[EmailEnvelope]] = {
        "urgent": [],
        "action_required": [],
        "newsletter": [],
        "fyi": [],
    }
    for env in envelopes:
        cat = categorize(env)
        result[cat].append(env)
    return result


def filter_by_category(
    envelopes: list[EmailEnvelope],
    category: str,
) -> list[EmailEnvelope]:
    """Return only emails that triage to *category*.

    This powers queries like "show urgent emails" — only emails that
    :func:`categorize` assigns to *category* are returned.

    Raises :class:`ValueError` for unknown category names.
    """
    if category not in TRIAGE_CATEGORIES:
        raise ValueError(
            f"Unknown triage category {category!r}. "
            f"Valid values: {', '.join(TRIAGE_CATEGORIES)}"
        )
    return [e for e in envelopes if categorize(e) == category]


__all__ = [
    "TRIAGE_CATEGORIES",
    "categorize",
    "filter_by_category",
    "triage_emails",
]
