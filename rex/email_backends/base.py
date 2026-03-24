"""Base interface for email backends.

All email backends (stub, IMAP/SMTP) must implement this interface so that
the EmailService can swap backends transparently.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class EmailEnvelope:
    """Lightweight representation of an email fetched from a backend.

    This is the transport-level object returned by backends.  The higher-level
    ``EmailSummary`` Pydantic model used by the service layer is built *from*
    this dataclass so that backends stay free of Pydantic dependencies.
    """

    message_id: str
    from_addr: str
    subject: str
    snippet: str
    received_at: datetime
    to_addrs: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SendResult:
    """Outcome of an email send operation."""

    ok: bool
    message_id: str | None = None
    error: str | None = None


class EmailBackend(abc.ABC):
    """Abstract base class for email read/send backends.

    Concrete implementations:
    - ``StubEmailBackend``: reads from a JSON fixture; send is a no-op.
    - ``ImapSmtpEmailBackend``: real IMAP4-SSL read + SMTP send.
    """

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def connect(self) -> bool:
        """Establish a connection to the email provider."""

    @abc.abstractmethod
    def fetch_unread(self, limit: int = 10) -> list[EmailEnvelope]:
        """Return up to *limit* unread email envelopes."""

    @abc.abstractmethod
    def list_mailboxes(self) -> list[str]:
        """Return the names of all available mailboxes / folders."""

    @abc.abstractmethod
    def mark_as_read(self, message_id: str) -> bool:
        """Flag *message_id* as read / seen."""

    # ------------------------------------------------------------------
    # Send operations
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def send(
        self,
        *,
        from_addr: str,
        to_addrs: list[str],
        subject: str,
        body: str,
        reply_to: str | None = None,
    ) -> SendResult:
        """Send an email."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def disconnect(self) -> None:  # noqa: B027
        """Release any held resources (connections, sockets)."""

    @property
    def is_connected(self) -> bool:  # noqa: D401
        """Whether the backend currently holds an active connection."""
        return False


__all__ = [
    "EmailBackend",
    "EmailEnvelope",
    "SendResult",
]
