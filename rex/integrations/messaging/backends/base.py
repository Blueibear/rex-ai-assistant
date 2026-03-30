"""Transport-layer ABC for SMS/messaging backends.

All SMS backends must implement this interface so that SMSService can swap them
transparently between stub and Twilio (and future) backends.
"""

from __future__ import annotations

import abc


class SMSBackend(abc.ABC):
    """Abstract base class for SMS send/receive backends.

    Methods return plain :class:`dict` objects so that backends remain free of
    higher-level model dependencies.

    Concrete implementations:
    - ``StubSMSBackend``: in-memory stub for offline development and testing.
    - ``TwilioSMSBackend``: real Twilio REST API backend (US-211).
    """

    @abc.abstractmethod
    def send(self, to: str, body: str) -> None:
        """Send an SMS message.

        Args:
            to:   Recipient phone number in E.164 format (e.g. ``"+15550001234"``).
            body: Message body text.

        Raises:
            Exception: On send failure (subclasses may raise specific errors).
        """

    @abc.abstractmethod
    def receive(self) -> list[dict]:
        """Return inbound messages not yet acknowledged by the application.

        Each dict must include at minimum:
        ``id``, ``from``, ``body``, ``received_at``.

        Returns:
            List of message dicts, newest first.
        """


__all__ = ["SMSBackend"]
