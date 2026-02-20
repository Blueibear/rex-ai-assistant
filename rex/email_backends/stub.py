"""Stub email backend backed by a local JSON fixture.

This backend is the default for offline development and testing.  It reads
mock email data from a JSON file and treats ``send()`` as a logged no-op.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rex.email_backends.base import EmailBackend, EmailEnvelope, SendResult

logger = logging.getLogger(__name__)

_DEFAULT_FIXTURE = Path("data/mock_emails.json")


class StubEmailBackend(EmailBackend):
    """JSON-fixture email backend for offline development and tests."""

    def __init__(self, fixture_path: Optional[Path] = None) -> None:
        self._fixture_path = fixture_path or _DEFAULT_FIXTURE
        self._emails: list[EmailEnvelope] = []
        self._connected = False
        self._sent: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        try:
            self._load_fixture()
            self._connected = True
            logger.info("StubEmailBackend connected (fixture: %s)", self._fixture_path)
            return True
        except Exception as exc:
            logger.error("StubEmailBackend failed to connect: %s", exc)
            self._connected = False
            return False

    def fetch_unread(self, limit: int = 10) -> list[EmailEnvelope]:
        if not self._connected:
            if not self.connect():
                return []
        unread = [e for e in self._emails if "unread" in e.labels]
        return unread[:max(0, limit)]

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
        reply_to: Optional[str] = None,
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
        return SendResult(ok=True, message_id="stub-msg-id")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_fixture(self) -> None:
        if not self._fixture_path.exists():
            logger.warning("Stub fixture not found at %s", self._fixture_path)
            self._emails = []
            return

        raw = self._fixture_path.read_text(encoding="utf-8")
        data = json.loads(raw)

        items: list[dict[str, Any]]
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and isinstance(data.get("messages"), list):
            items = [x for x in data["messages"] if isinstance(x, dict)]
        else:
            logger.warning("Unrecognised fixture format in %s", self._fixture_path)
            self._emails = []
            return

        parsed: list[EmailEnvelope] = []
        for item in items:
            try:
                parsed.append(self._dict_to_envelope(item))
            except Exception as exc:
                logger.warning("Skipping invalid fixture record: %s", exc)

        self._emails = parsed
        logger.info("StubEmailBackend loaded %d messages", len(parsed))

    @staticmethod
    def _dict_to_envelope(d: dict[str, Any]) -> EmailEnvelope:
        msg_id = str(d.get("id") or d.get("message_id") or "")
        if not msg_id:
            raise ValueError("missing id/message_id")

        from_addr = str(d.get("from_addr") or d.get("sender") or "unknown@example.com")
        subject = str(d.get("subject") or "")
        snippet = d.get("snippet") or str(d.get("body", ""))[:200]

        received_raw = d.get("received_at")
        received_at = _parse_dt(received_raw) or datetime.now(timezone.utc)

        labels = d.get("labels")
        if not isinstance(labels, list):
            labels = ["unread"]

        return EmailEnvelope(
            message_id=msg_id,
            from_addr=from_addr,
            subject=subject,
            snippet=str(snippet),
            received_at=received_at,
            labels=[str(lb) for lb in labels],
        )

    @property
    def sent_messages(self) -> list[dict[str, Any]]:
        """Accessor for test assertions on messages 'sent' via stub."""
        return list(self._sent)


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


__all__ = ["StubEmailBackend"]
