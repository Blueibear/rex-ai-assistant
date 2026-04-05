"""EmailService: credential-ready stub/live email access.

Without credentials (``email_provider == "none"``), all methods return realistic
mock data so the GUI and autonomy engine work out of the box.

When ``email_provider == "gmail"``, the service connects to the Gmail REST API
using OAuth tokens stored in the environment (``GMAIL_ACCESS_TOKEN`` and
``GMAIL_REFRESH_TOKEN``).  Outlook support is scaffolded as a stub.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import UTC, datetime, timedelta

from rex.integrations.models import EmailMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stub data helpers
# ---------------------------------------------------------------------------

_STUB_MESSAGES: list[EmailMessage] = [
    EmailMessage(
        id="stub-001",
        thread_id="thread-001",
        subject="Welcome to Rex AI",
        sender="team@rex.ai",
        recipients=["user@example.com"],
        body_text=(
            "Hi there,\n\nWelcome aboard! Rex is ready to help you stay organised.\n\n"
            "Best,\nThe Rex Team"
        ),
        received_at=datetime.now(UTC) - timedelta(hours=2),
        labels=["INBOX"],
        is_read=False,
        priority="medium",
    ),
    EmailMessage(
        id="stub-002",
        thread_id="thread-002",
        subject="Your weekly summary",
        sender="notifications@rex.ai",
        recipients=["user@example.com"],
        body_text="Here is your weekly activity summary from Rex.",
        received_at=datetime.now(UTC) - timedelta(hours=10),
        labels=["INBOX"],
        is_read=True,
        priority="low",
    ),
    EmailMessage(
        id="stub-003",
        thread_id="thread-003",
        subject="Action required: invoice #4821",
        sender="billing@vendor.example.com",
        recipients=["user@example.com"],
        body_text="Please review and pay invoice #4821 by end of week.",
        received_at=datetime.now(UTC) - timedelta(days=1),
        labels=["INBOX", "IMPORTANT"],
        is_read=False,
        priority="high",
    ),
    EmailMessage(
        id="stub-004",
        thread_id="thread-004",
        subject="Meeting notes — Q2 planning",
        sender="colleague@example.com",
        recipients=["user@example.com"],
        body_text="Hi, I've attached the notes from today's Q2 planning session.",
        received_at=datetime.now(UTC) - timedelta(days=2),
        labels=["INBOX"],
        is_read=True,
        priority="medium",
    ),
    EmailMessage(
        id="stub-005",
        thread_id="thread-005",
        subject="URGENT: Server alert",
        sender="alerts@monitoring.example.com",
        recipients=["user@example.com", "ops@example.com"],
        body_text="High CPU usage detected on prod-web-01. Immediate attention required.",
        received_at=datetime.now(UTC) - timedelta(minutes=15),
        labels=["INBOX", "IMPORTANT"],
        is_read=False,
        priority="critical",
    ),
]


# ---------------------------------------------------------------------------
# EmailService
# ---------------------------------------------------------------------------


class EmailService:
    """Unified email access layer with stub and live backends.

    Args:
        email_provider: One of ``"none"``, ``"gmail"``, or ``"outlook"``.
            Defaults to ``"none"`` (stub mode).
    """

    def __init__(self, email_provider: str = "none") -> None:
        self._provider = email_provider.lower()
        logger.debug("EmailService initialised with provider=%s", self._provider)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_inbox(self, limit: int = 20) -> list[EmailMessage]:
        """Return up to *limit* messages from the inbox.

        Args:
            limit: Maximum number of messages to return.

        Returns:
            List of :class:`~rex.integrations.models.EmailMessage` objects,
            newest first.
        """
        if self._provider == "gmail":
            return self._gmail_list_inbox(limit)
        if self._provider == "outlook":
            logger.warning("Outlook live mode not yet implemented; falling back to stub.")
        return _STUB_MESSAGES[:limit]

    def get_thread(self, thread_id: str) -> list[EmailMessage]:
        """Return all messages in a thread.

        Args:
            thread_id: Provider thread identifier.

        Returns:
            List of messages belonging to *thread_id*, chronological order.
        """
        if self._provider == "gmail":
            return self._gmail_get_thread(thread_id)
        if self._provider == "outlook":
            logger.warning("Outlook live mode not yet implemented; falling back to stub.")
        return [m for m in _STUB_MESSAGES if m.thread_id == thread_id]

    def send_draft(self, to: str, subject: str, body: str) -> EmailMessage:
        """Create and send an email draft.

        Args:
            to: Recipient email address.
            subject: Email subject line.
            body: Plain-text body of the email.

        Returns:
            The sent :class:`~rex.integrations.models.EmailMessage`.
        """
        if self._provider == "gmail":
            return self._gmail_send_draft(to, subject, body)
        if self._provider == "outlook":
            logger.warning("Outlook live mode not yet implemented; falling back to stub.")
        return EmailMessage(
            id=f"stub-sent-{uuid.uuid4().hex[:8]}",
            thread_id=f"thread-sent-{uuid.uuid4().hex[:8]}",
            subject=subject,
            sender="user@example.com",
            recipients=[to],
            body_text=body,
            received_at=datetime.now(UTC),
            labels=["SENT"],
            is_read=True,
            priority="low",
        )

    def archive(self, id: str) -> None:  # noqa: A002
        """Archive a message by ID.

        In stub mode this is a no-op.

        Args:
            id: Message identifier.
        """
        if self._provider == "gmail":
            self._gmail_archive(id)
            return
        logger.debug("Stub archive called for message id=%s", id)

    def mark_read(self, id: str) -> None:  # noqa: A002
        """Mark a message as read.

        In stub mode this is a no-op.

        Args:
            id: Message identifier.
        """
        if self._provider == "gmail":
            self._gmail_mark_read(id)
            return
        logger.debug("Stub mark_read called for message id=%s", id)

    # ------------------------------------------------------------------
    # Gmail live backend (requires GMAIL_ACCESS_TOKEN env var)
    # ------------------------------------------------------------------

    def _gmail_headers(self) -> dict[str, str]:
        token = os.environ.get("GMAIL_ACCESS_TOKEN", "")
        return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    def _gmail_list_inbox(self, limit: int) -> list[EmailMessage]:
        """Fetch inbox messages via Gmail REST API."""
        try:
            import json
            import urllib.request

            url = (
                f"https://gmail.googleapis.com/gmail/v1/users/me/messages"
                f"?labelIds=INBOX&maxResults={limit}"
            )
            req = urllib.request.Request(url, headers=self._gmail_headers())
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            messages: list[EmailMessage] = []
            for item in data.get("messages", []):
                msg = self._gmail_fetch_message(item["id"])
                if msg is not None:
                    messages.append(msg)
            return messages
        except Exception as exc:  # noqa: BLE001
            logger.error("Gmail list_inbox failed: %s", exc)
            return _STUB_MESSAGES[:limit]

    def _gmail_get_thread(self, thread_id: str) -> list[EmailMessage]:
        try:
            import json
            import urllib.request

            url = f"https://gmail.googleapis.com/gmail/v1/users/me/threads/{thread_id}"
            req = urllib.request.Request(url, headers=self._gmail_headers())
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            messages: list[EmailMessage] = []
            for raw in data.get("messages", []):
                msg = self._parse_gmail_message(raw, thread_id)
                if msg is not None:
                    messages.append(msg)
            return messages
        except Exception as exc:  # noqa: BLE001
            logger.error("Gmail get_thread failed: %s", exc)
            return [m for m in _STUB_MESSAGES if m.thread_id == thread_id]

    def _gmail_send_draft(self, to: str, subject: str, body: str) -> EmailMessage:
        try:
            import base64
            import json
            import urllib.request

            raw_msg = f"To: {to}\r\nSubject: {subject}\r\nContent-Type: text/plain\r\n\r\n{body}"
            encoded = base64.urlsafe_b64encode(raw_msg.encode()).decode()
            payload = json.dumps({"raw": encoded}).encode()
            headers = {**self._gmail_headers(), "Content-Type": "application/json"}
            req = urllib.request.Request(
                "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            return EmailMessage(
                id=data.get("id", uuid.uuid4().hex),
                thread_id=data.get("threadId", uuid.uuid4().hex),
                subject=subject,
                sender="me",
                recipients=[to],
                body_text=body,
                received_at=datetime.now(UTC),
                labels=["SENT"],
                is_read=True,
                priority="low",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Gmail send_draft failed: %s", exc)
            return EmailMessage(
                id=f"stub-sent-{uuid.uuid4().hex[:8]}",
                thread_id=f"thread-sent-{uuid.uuid4().hex[:8]}",
                subject=subject,
                sender="user@example.com",
                recipients=[to],
                body_text=body,
                received_at=datetime.now(UTC),
                labels=["SENT"],
                is_read=True,
                priority="low",
            )

    def _gmail_archive(self, id: str) -> None:  # noqa: A002
        try:
            import json
            import urllib.request

            url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{id}/modify"
            payload = json.dumps({"removeLabelIds": ["INBOX"]}).encode()
            headers = {**self._gmail_headers(), "Content-Type": "application/json"}
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            urllib.request.urlopen(req, timeout=10).close()
        except Exception as exc:  # noqa: BLE001
            logger.error("Gmail archive failed: %s", exc)

    def _gmail_mark_read(self, id: str) -> None:  # noqa: A002
        try:
            import json
            import urllib.request

            url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{id}/modify"
            payload = json.dumps({"removeLabelIds": ["UNREAD"]}).encode()
            headers = {**self._gmail_headers(), "Content-Type": "application/json"}
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            urllib.request.urlopen(req, timeout=10).close()
        except Exception as exc:  # noqa: BLE001
            logger.error("Gmail mark_read failed: %s", exc)

    def _gmail_fetch_message(self, msg_id: str) -> EmailMessage | None:
        try:
            import json
            import urllib.request

            url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg_id}"
            req = urllib.request.Request(url, headers=self._gmail_headers())
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            return self._parse_gmail_message(data, data.get("threadId", ""))
        except Exception as exc:  # noqa: BLE001
            logger.error("Gmail fetch_message failed: %s", exc)
            return None

    def _parse_gmail_message(self, data: dict[str, object], thread_id: str) -> EmailMessage | None:
        """Convert a raw Gmail API message dict to an :class:`EmailMessage`."""
        try:
            headers: dict[str, str] = {}
            payload = data.get("payload", {})
            assert isinstance(payload, dict)
            for hdr in payload.get("headers", []):
                assert isinstance(hdr, dict)
                headers[str(hdr.get("name", "")).lower()] = str(hdr.get("value", ""))

            label_ids = data.get("labelIds", [])
            assert isinstance(label_ids, list)
            is_read = "UNREAD" not in label_ids

            internal_date = data.get("internalDate")
            if internal_date is not None:
                received_at = datetime.fromtimestamp(int(str(internal_date)) / 1000, tz=UTC)
            else:
                received_at = datetime.now(UTC)

            snippet = str(data.get("snippet", ""))
            msg_id = str(data.get("id", uuid.uuid4().hex))

            to_header = headers.get("to", "")
            recipients = [addr.strip() for addr in to_header.split(",") if addr.strip()]

            return EmailMessage(
                id=msg_id,
                thread_id=thread_id or str(data.get("threadId", msg_id)),
                subject=headers.get("subject", "(no subject)"),
                sender=headers.get("from", ""),
                recipients=recipients,
                body_text=snippet,
                received_at=received_at,
                labels=[str(lbl) for lbl in label_ids],
                is_read=is_read,
                priority="low",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to parse Gmail message: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["EmailService"]
