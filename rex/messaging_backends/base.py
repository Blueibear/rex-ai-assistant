"""Base interface for SMS/messaging backends.

All SMS backends (stub, Twilio) must implement this interface so that
the messaging service can swap backends transparently.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class SmsSendResult:
    """Outcome of an SMS send operation."""

    ok: bool
    message_sid: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class InboundSms:
    """Lightweight representation of an inbound SMS message."""

    sid: str
    from_number: str
    to_number: str
    body: str
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SmsBackend(abc.ABC):
    """Abstract base class for SMS send/receive backends.

    Concrete implementations:
    - ``StubSmsBackend``: writes to a local JSON file; no real delivery.
    - ``TwilioSmsBackend``: uses the Twilio REST API for real delivery.
    """

    @abc.abstractmethod
    def send_sms(
        self,
        *,
        to: str,
        body: str,
        from_number: str | None = None,
    ) -> SmsSendResult:
        """Send an SMS message.

        Args:
            to: Destination phone number (E.164 format).
            body: Message body text.
            from_number: Override sender number (optional).

        Returns:
            A ``SmsSendResult`` indicating success or failure.
        """

    @abc.abstractmethod
    def fetch_recent_inbound(self, *, limit: int = 20) -> list[InboundSms]:
        """Retrieve recent inbound SMS messages.

        Args:
            limit: Maximum number of messages to return.

        Returns:
            List of inbound messages, newest first.
        """

    def disconnect(self) -> None:  # noqa: B027
        """Release any held resources."""


__all__ = [
    "InboundSms",
    "SmsBackend",
    "SmsSendResult",
]
