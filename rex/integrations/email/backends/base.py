"""Transport-layer ABC for email backends.

All email backends must implement this interface so that EmailService can swap
them transparently between stub, IMAP/SMTP, and future OAuth backends.
"""

from __future__ import annotations

import abc


class EmailBackend(abc.ABC):
    """Abstract base class for email read/send backends.

    Methods return plain :class:`dict` objects so that backends remain free of
    higher-level model dependencies.

    Concrete implementations:
    - ``StubEmailBackend``: in-memory stub for offline development and testing.
    - ``IMAPSMTPBackend``: real IMAP4-SSL read + SMTP send (US-206/207).
    """

    @abc.abstractmethod
    def fetch_unread(self, limit: int = 10) -> list[dict]:
        """Return up to *limit* unread messages as plain dicts.

        Each dict must include at minimum:
        ``id``, ``from``, ``subject``, ``snippet``, ``received_at``.

        Args:
            limit: Maximum number of messages to return.

        Returns:
            List of message dicts, newest first.
        """

    @abc.abstractmethod
    def send(self, to: str, subject: str, body: str) -> None:
        """Send an email message.

        Args:
            to:      Recipient address.
            subject: Email subject line.
            body:    Plain-text message body.

        Raises:
            Exception: On send failure (subclasses may raise specific errors).
        """


__all__ = ["EmailBackend"]
