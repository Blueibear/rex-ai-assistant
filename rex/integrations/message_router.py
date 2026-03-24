"""MessageRouter: dispatch outbound messages to SMS or email.

Routes a message to the correct channel based on an explicit
``preferred_channel`` argument, a contacts file
(``~/.rex/contacts.json``), or availability of configured services.

Priority:
1. ``preferred_channel`` parameter — used directly if supplied.
2. Contact entry in ``~/.rex/contacts.json`` — uses the stored
   ``preferred_channel`` for that contact (matched by name or number).
3. Service availability — email if :class:`EmailService` is configured;
   SMS if :class:`SMSService` is configured; raises :exc:`NoChannelError`
   if neither.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class NoChannelError(Exception):
    """Raised when no delivery channel is available for the contact."""


# ---------------------------------------------------------------------------
# MessageResult model
# ---------------------------------------------------------------------------


class MessageResult(BaseModel):
    """Result of a routed message dispatch.

    Attributes:
        channel: The channel used (``"sms"`` or ``"email"``).
        message_id: Provider-assigned or stub message identifier.
        status: Delivery status string (e.g. ``"sent"``, ``"stub"``).
    """

    channel: str
    message_id: str
    status: str


# ---------------------------------------------------------------------------
# Channel protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class SMSChannel(Protocol):
    """Minimal interface required from an SMS service."""

    def send(self, to: str, body: str) -> object:
        """Send an SMS; returned object must have .id and .status attributes."""
        ...


@runtime_checkable
class EmailChannel(Protocol):
    """Minimal interface required from an email service."""

    def send_draft(self, to: str, subject: str, body: str) -> object:
        """Send an email draft; returned object must have .id attribute."""
        ...


# ---------------------------------------------------------------------------
# Contacts helper
# ---------------------------------------------------------------------------

_CONTACTS_PATH = Path.home() / ".rex" / "contacts.json"


def _load_contacts() -> list[dict[str, str]]:
    """Load contacts from ``~/.rex/contacts.json`` (returns empty list on error)."""
    try:
        if _CONTACTS_PATH.exists():
            raw = _CONTACTS_PATH.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, list):
                return [c for c in data if isinstance(c, dict)]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load contacts file: %s", exc)
    return []


def _lookup_preferred_channel(contact: str) -> str | None:
    """Return the stored preferred channel for *contact*, or ``None``."""
    contacts = _load_contacts()
    contact_lower = contact.lower()
    for entry in contacts:
        name = str(entry.get("name", "")).lower()
        number = str(entry.get("number", "")).lower()
        if contact_lower in (name, number):
            ch = entry.get("preferred_channel", "")
            if ch in ("sms", "email"):
                return ch
    return None


# ---------------------------------------------------------------------------
# MessageRouter
# ---------------------------------------------------------------------------


class MessageRouter:
    """Route outbound messages to SMS or email.

    Args:
        sms: Optional :class:`SMSChannel` implementation.  When ``None``
            the SMS channel is unavailable.
        email: Optional :class:`EmailChannel` implementation.  When
            ``None`` the email channel is unavailable.
    """

    def __init__(
        self,
        sms: SMSChannel | None = None,
        email: EmailChannel | None = None,
    ) -> None:
        self._sms = sms
        self._email = email

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        contact: str,
        body: str,
        preferred_channel: str | None = None,
    ) -> MessageResult:
        """Dispatch *body* to *contact* via the best available channel.

        Args:
            contact: Recipient identifier — an email address, a phone
                number in E.164 format, or a display name that can be
                looked up in ``~/.rex/contacts.json``.
            body: Message body text.
            preferred_channel: Explicit channel override (``"sms"`` or
                ``"email"``).  When supplied, skips all routing logic.

        Returns:
            A :class:`MessageResult` describing the outcome.

        Raises:
            NoChannelError: When no channel is configured or the
                requested channel is unavailable.
            ValueError: When *preferred_channel* is an unknown value.
        """
        channel = self._resolve_channel(contact, preferred_channel)
        if channel == "sms":
            return self._send_sms(contact, body)
        return self._send_email(contact, body)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_channel(self, contact: str, preferred_channel: str | None) -> str:
        if preferred_channel is not None:
            ch = preferred_channel.lower()
            if ch not in ("sms", "email"):
                raise ValueError(f"Unknown channel: {preferred_channel!r}")
            if ch == "sms" and self._sms is None:
                raise NoChannelError("SMS channel requested but SMSService not configured.")
            if ch == "email" and self._email is None:
                raise NoChannelError("Email channel requested but EmailService not configured.")
            return ch

        # Try contacts lookup
        stored = _lookup_preferred_channel(contact)
        if stored == "sms" and self._sms is not None:
            return "sms"
        if stored == "email" and self._email is not None:
            return "email"

        # Fall back to availability
        if self._email is not None:
            return "email"
        if self._sms is not None:
            return "sms"

        raise NoChannelError(f"No channel configured to reach contact {contact!r}.")

    def _send_sms(self, to: str, body: str) -> MessageResult:
        assert self._sms is not None
        try:
            result = self._sms.send(to, body)
            msg_id = str(getattr(result, "id", "unknown"))
            status = str(getattr(result, "status", "sent"))
            return MessageResult(channel="sms", message_id=msg_id, status=status)
        except Exception as exc:  # noqa: BLE001
            logger.error("SMS send failed: %s", exc)
            raise

    def _send_email(self, to: str, body: str) -> MessageResult:
        assert self._email is not None
        subject = body[:50] + ("…" if len(body) > 50 else "")
        try:
            result = self._email.send_draft(to, subject, body)
            msg_id = str(getattr(result, "id", "unknown"))
            return MessageResult(channel="email", message_id=msg_id, status="sent")
        except Exception as exc:  # noqa: BLE001
            logger.error("Email send failed: %s", exc)
            raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "EmailChannel",
    "MessageResult",
    "MessageRouter",
    "NoChannelError",
    "SMSChannel",
]
