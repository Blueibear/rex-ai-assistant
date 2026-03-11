"""EmailInboxStub — in-memory email inbox stub with built-in mock data.

This module provides ``EmailInboxStub``, a zero-dependency email backend
whose mock data is defined in code rather than loaded from a fixture file.
It is designed for unit tests and beta-phase triage feature development where
no live credentials or network access is available.

Mock emails cover at least three triage categories:
  - ``urgent``          — time-sensitive / high-priority messages
  - ``action_required`` — messages that need a response or action
  - ``fyi``             — informational messages requiring no action

The category is stored as a label on ``EmailEnvelope.labels`` so that triage
logic can read it without knowing the stub internals.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from rex.email_backends.base import EmailBackend, EmailEnvelope, SendResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in mock emails
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 11, 9, 0, 0, tzinfo=timezone.utc)


def _dt(day: int, hour: int = 9) -> datetime:
    return _NOW.replace(day=day, hour=hour)


_MOCK_EMAILS: list[EmailEnvelope] = [
    # ── urgent ──────────────────────────────────────────────────────────────
    EmailEnvelope(
        message_id="mock-urgent-001",
        from_addr="cto@example.com",
        subject="URGENT: Production outage — all hands",
        snippet="Our payment service is down. Please join the incident bridge immediately.",
        received_at=_dt(11, 6),
        to_addrs=["team@example.com"],
        labels=["unread", "urgent"],
    ),
    EmailEnvelope(
        message_id="mock-urgent-002",
        from_addr="alerts@monitor.example.com",
        subject="CRITICAL: Disk usage at 98% on prod-db-01",
        snippet="Disk usage has exceeded the 95% threshold. Immediate action required.",
        received_at=_dt(11, 7),
        to_addrs=["ops@example.com"],
        labels=["unread", "urgent"],
    ),
    # ── action_required ─────────────────────────────────────────────────────
    EmailEnvelope(
        message_id="mock-action-001",
        from_addr="hr@example.com",
        subject="Action required: please complete your annual review by Friday",
        snippet="Your annual self-review form is due this Friday. Please log in to complete it.",
        received_at=_dt(10, 14),
        to_addrs=["user@example.com"],
        labels=["unread", "action_required"],
    ),
    EmailEnvelope(
        message_id="mock-action-002",
        from_addr="billing@saas-vendor.com",
        subject="Invoice #INV-20260310 due in 7 days",
        snippet="Your subscription invoice of $299 is due on March 18. Please update payment.",
        received_at=_dt(10, 11),
        to_addrs=["accounts@example.com"],
        labels=["unread", "action_required"],
    ),
    # ── fyi ─────────────────────────────────────────────────────────────────
    EmailEnvelope(
        message_id="mock-fyi-001",
        from_addr="newsletter@techdigest.example.com",
        subject="This week in AI — March 2026 edition",
        snippet="Top stories: new open-weight models, AGI timeline debate, and more.",
        received_at=_dt(9, 8),
        to_addrs=["user@example.com"],
        labels=["fyi"],
    ),
    EmailEnvelope(
        message_id="mock-fyi-002",
        from_addr="noreply@github.com",
        subject="Your pull request was merged",
        snippet="rex-ai-assistant #205 has been merged into master by james.",
        received_at=_dt(9, 16),
        to_addrs=["user@example.com"],
        labels=["fyi"],
    ),
    EmailEnvelope(
        message_id="mock-fyi-003",
        from_addr="calendar-noreply@example.com",
        subject="Reminder: team stand-up tomorrow at 09:00",
        snippet="Just a friendly reminder about tomorrow's stand-up.",
        received_at=_dt(10, 17),
        to_addrs=["team@example.com"],
        labels=["fyi"],
    ),
]


# ---------------------------------------------------------------------------
# EmailInboxStub
# ---------------------------------------------------------------------------


class EmailInboxStub(EmailBackend):
    """In-memory email backend with built-in mock data.

    Designed for offline unit tests and beta-phase triage features.
    All reads return data from ``_MOCK_EMAILS``; sends are no-ops that record
    the outbound message in ``sent_messages``.

    The stub implements the same ``EmailBackend`` interface as the real IMAP/SMTP
    backend so that calling code needs no changes when switching backends.
    """

    def __init__(self) -> None:
        self._emails: list[EmailEnvelope] = list(_MOCK_EMAILS)
        self._sent: list[dict[str, Any]] = []
        self._connected = False

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        self._connected = True
        logger.info("EmailInboxStub connected (using built-in mock data)")
        return True

    def fetch_unread(self, limit: int = 10) -> list[EmailEnvelope]:
        if not self._connected:
            self.connect()
        unread = [e for e in self._emails if "unread" in e.labels]
        return unread[: max(0, limit)]

    def list_mailboxes(self) -> list[str]:
        return ["INBOX"]

    def mark_as_read(self, message_id: str) -> bool:
        for i, env in enumerate(self._emails):
            if env.message_id == message_id and "unread" in env.labels:
                new_labels = [lb for lb in env.labels if lb != "unread"]
                self._emails[i] = EmailEnvelope(
                    message_id=env.message_id,
                    from_addr=env.from_addr,
                    subject=env.subject,
                    snippet=env.snippet,
                    received_at=env.received_at,
                    to_addrs=env.to_addrs,
                    labels=new_labels,
                )
                return True
        return False

    def fetch_by_category(self, category: str) -> list[EmailEnvelope]:
        """Return all emails whose labels include *category*."""
        return [e for e in self._emails if category in e.labels]

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    def send(
        self,
        *,
        from_addr: str,
        to_addrs: list[str],
        subject: str,
        body: str,
        reply_to: str | None = None,
    ) -> SendResult:
        logger.info(
            "[STUB] Would send email from=%s to=%s subject=%r",
            from_addr,
            to_addrs,
            subject,
        )
        self._sent.append(
            {
                "from_addr": from_addr,
                "to_addrs": to_addrs,
                "subject": subject,
                "body": body,
                "reply_to": reply_to,
            }
        )
        return SendResult(ok=True, message_id="stub-inbox-msg-id")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    @property
    def sent_messages(self) -> list[dict[str, Any]]:
        """Messages 'sent' during this session — useful for test assertions."""
        return list(self._sent)

    @property
    def all_emails(self) -> list[EmailEnvelope]:
        """All mock emails (read + unread)."""
        return list(self._emails)


__all__ = ["EmailInboxStub"]
