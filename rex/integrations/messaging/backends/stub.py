"""Stub SMS backend for offline development and testing.

Records ``send()`` calls as a no-op and returns fixed in-memory messages for
``receive()``.  Implements the :class:`SMSBackend` transport-layer interface.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from rex.integrations.messaging.backends.base import SMSBackend

logger = logging.getLogger(__name__)

_STUB_INBOUND: list[dict] = [
    {
        "id": "stub-sms-001",
        "from": "+14155550101",
        "body": "Hey, are you free for a call tomorrow?",
        "received_at": "2026-03-28T08:30:00+00:00",
    },
    {
        "id": "stub-sms-002",
        "from": "+14155550202",
        "body": "Don't forget the team lunch at noon.",
        "received_at": "2026-03-28T07:00:00+00:00",
    },
]


class StubSMSBackend(SMSBackend):
    """In-memory stub SMS backend implementing the transport-layer interface."""

    def __init__(self, inbound: list[dict] | None = None) -> None:
        self._inbound: list[dict] = list(inbound or _STUB_INBOUND)
        self._sent: list[dict] = []

    def send(self, to: str, body: str) -> None:
        """Log the send and record it for test assertions."""
        logger.info("[STUB SMS] To=%s Body=%r", to, body)
        self._sent.append(
            {
                "id": f"stub-sms-out-{uuid.uuid4().hex[:8]}",
                "to": to,
                "body": body,
                "sent_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def receive(self) -> list[dict]:
        """Return stub inbound messages."""
        return list(self._inbound)

    @property
    def sent_messages(self) -> list[dict]:
        """Messages recorded by ``send()`` — useful for test assertions."""
        return list(self._sent)


__all__ = ["StubSMSBackend"]
