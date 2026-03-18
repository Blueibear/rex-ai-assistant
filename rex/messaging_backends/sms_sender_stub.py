"""In-memory SMS sender stub for offline development and testing.

``SmsSenderStub`` accepts a phone number and message body and stores sent
messages in a structured in-memory log.  No network calls are ever made.

The ``send`` method signature is intentionally compatible with the
``TwilioAdapter`` interface defined in US-086 so that the stub can be
swapped for a real Twilio client without changing calling code.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SentSmsRecord:
    """Structured record of a single outbound SMS."""

    sid: str
    to: str
    body: str
    sent_at: datetime


class SmsSenderStub:
    """In-memory SMS send stub.

    Sent messages are stored in ``self._log`` and exposed via the
    ``sent_messages`` property for test assertions.  No real network
    calls are ever made.
    """

    def __init__(self, default_from: str = "+15555551234") -> None:
        self._default_from = default_from
        self._log: list[SentSmsRecord] = []

    # ------------------------------------------------------------------
    # Primary interface (matches TwilioAdapter — US-086)
    # ------------------------------------------------------------------

    def send(self, to: str, body: str) -> dict[str, Any]:
        """Send (stub) an SMS message.

        Args:
            to: Destination phone number (E.164 format preferred).
            body: Message body text.

        Returns:
            A dict with keys ``ok`` (bool), ``sid`` (str), and
            ``to`` / ``body`` echoed back.
        """
        sid = f"stub_{uuid.uuid4().hex[:16]}"
        record = SentSmsRecord(
            sid=sid,
            to=to,
            body=body,
            sent_at=datetime.now(timezone.utc),
        )
        self._log.append(record)
        logger.info("[SmsSenderStub] Would send SMS to %s: %s", to, body[:80])
        return {"ok": True, "sid": sid, "to": to, "body": body}

    # ------------------------------------------------------------------
    # TwilioAdapter interface (US-086)
    # ------------------------------------------------------------------

    def send_sms(self, to: str, body: str) -> dict[str, Any]:
        """Implement ``TwilioAdapter.send_sms``.

        Delegates to :meth:`send` so that callers using the Protocol interface
        get the same stub behaviour as direct ``send`` callers.

        Args:
            to: Destination phone number (E.164 format preferred).
            body: Message body text.

        Returns:
            The same dict returned by :meth:`send`.
        """
        return self.send(to, body)

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    @property
    def sent_messages(self) -> list[SentSmsRecord]:
        """Return a copy of the in-memory outbound message log."""
        return list(self._log)

    def clear(self) -> None:
        """Clear the in-memory log (useful between test cases)."""
        self._log.clear()


__all__ = ["SentSmsRecord", "SmsSenderStub"]
