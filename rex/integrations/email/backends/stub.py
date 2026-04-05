"""Stub email backend for offline development and testing.

Returns fixed in-memory messages for ``fetch_unread`` and logs send calls
as a no-op.  Implements the :class:`EmailBackend` transport-layer interface.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from rex.integrations.email.backends.base import EmailBackend

logger = logging.getLogger(__name__)

_STUB_MESSAGES: list[dict] = [
    {
        "id": "stub-email-001",
        "from": "alice@example.com",
        "subject": "Team standup notes",
        "snippet": "Here are the notes from today's standup...",
        "received_at": "2026-03-28T09:00:00+00:00",
    },
    {
        "id": "stub-email-002",
        "from": "bob@example.com",
        "subject": "Lunch tomorrow?",
        "snippet": "Are you free for lunch tomorrow at noon?",
        "received_at": "2026-03-28T10:30:00+00:00",
    },
]


class StubEmailBackend(EmailBackend):
    """In-memory stub email backend implementing the transport-layer interface."""

    def __init__(self, messages: list[dict] | None = None) -> None:
        self._messages: list[dict] = list(messages or _STUB_MESSAGES)
        self._sent: list[dict] = []

    def fetch_unread(self, limit: int = 10) -> list[dict]:
        """Return up to *limit* stub messages."""
        return self._messages[: max(0, limit)]

    def send(self, to: str, subject: str, body: str) -> None:
        """Log the send and record it for test assertions."""
        logger.info("[STUB EMAIL] To=%s Subject=%r", to, subject)
        self._sent.append(
            {
                "to": to,
                "subject": subject,
                "body": body,
                "sent_at": datetime.now(UTC).isoformat(),
            }
        )

    @property
    def sent_messages(self) -> list[dict]:
        """Messages recorded by ``send()`` — useful for test assertions."""
        return list(self._sent)


__all__ = ["StubEmailBackend"]
