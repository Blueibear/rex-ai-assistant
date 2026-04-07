"""EmailInboxStub — in-memory email backend for unit tests.

This module provides ``EmailInboxStub``, a zero-dependency email backend
designed for unit tests and offline feature development.  It carries no
built-in mock data; callers must inject email fixtures via the ``emails``
constructor parameter.

The stub implements the same ``EmailBackend`` interface as the real IMAP/SMTP
backend so that calling code needs no changes when switching backends.
"""

from __future__ import annotations

import logging
from typing import Any

from rex.email_backends.base import EmailBackend, EmailEnvelope, SendResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EmailInboxStub
# ---------------------------------------------------------------------------


class EmailInboxStub(EmailBackend):
    """In-memory email backend for unit tests.

    Starts empty unless ``emails`` is provided.  Sends are no-ops that record
    the outbound message in ``sent_messages``.

    The stub implements the same ``EmailBackend`` interface as the real IMAP/SMTP
    backend so that calling code needs no changes when switching backends.
    """

    def __init__(self, emails: list[EmailEnvelope] | None = None) -> None:
        self._emails: list[EmailEnvelope] = list(emails) if emails is not None else []
        self._sent: list[dict[str, Any]] = []
        self._connected = False

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        self._connected = True
        logger.info("EmailInboxStub connected (%d emails loaded)", len(self._emails))
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
