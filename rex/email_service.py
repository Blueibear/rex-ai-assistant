"""
Email service module for Rex AI Assistant.

Provides email triage functionality with classification and summarization.

Design goals:
- Works in stub/mock mode (no real IMAP/SMTP yet).
- Supports both:
  1) "fetch_unread(...)" returning structured EmailSummary objects
  2) "triage_unread(...)" returning CLI-friendly dict summaries
- Optional EventBus publishing if rex.event_bus is available and provided.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Optional EventBus import for compatibility with earlier implementation
try:
    from rex.event_bus import EventBus  # type: ignore
except Exception:  # pragma: no cover
    EventBus = Any  # type: ignore

# Optional credentials manager import
try:
    from rex.credentials import get_credential_manager  # type: ignore
except Exception:  # pragma: no cover
    get_credential_manager = None  # type: ignore

# Optional Pydantic import (nice to have, not required at runtime)
try:
    from pydantic import BaseModel, ConfigDict, Field, field_serializer  # type: ignore

    class EmailSummary(BaseModel):
        """Summary of an email message."""

        model_config = ConfigDict()

        id: str = Field(..., description="Unique email identifier")
        from_addr: str = Field(..., description="Sender email address")
        subject: str = Field(..., description="Email subject")
        snippet: str = Field(..., description="Brief preview of email body")
        received_at: datetime = Field(..., description="When the email was received")
        labels: list[str] = Field(default_factory=list, description="Email labels/tags")
        importance_score: float = Field(default=0.5, description="Importance score (0.0-1.0)")
        category: Optional[str] = Field(default=None, description="Email category (important, promo, social, etc.)")

        @field_serializer("received_at", when_used="json")
        @classmethod
        def _serialize_received_at(cls, v: datetime, _info: object) -> str:
            return v.isoformat()

except Exception:  # pragma: no cover
    @dataclass
    class EmailSummary:  # type: ignore
        """Fallback EmailSummary when pydantic is not installed."""

        id: str
        from_addr: str
        subject: str
        snippet: str
        received_at: datetime
        labels: list[str]
        importance_score: float = 0.5
        category: Optional[str] = None


@dataclass(frozen=True)
class EmailMessage:
    """Legacy/compat message type used by older CLI/service code."""

    message_id: str
    sender: str
    subject: str
    body: str
    received_at: datetime

    def to_summary(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "subject": self.subject,
            "received_at": self.received_at.isoformat(),
            "body": self.body,
        }


class EmailService:
    """
    Email service for reading and categorizing emails.

    Currently uses stub implementation with mock data.
    Real IMAP/SMTP integration can be added later.
    """

    def __init__(
        self,
        mock_data_file: Optional[Path] = None,
        *,
        event_bus: Optional["EventBus"] = None,
        mock_messages: Optional[list[EmailMessage]] = None,
        mock_data_path: Optional[Path] = None,
    ) -> None:
        if event_bus is None and mock_data_file is not None and hasattr(mock_data_file, "publish"):
            event_bus = mock_data_file  # type: ignore[assignment]
            mock_data_file = None

        if mock_data_file is None and mock_data_path is not None:
            mock_data_file = mock_data_path

        self.mock_data_file = mock_data_file or Path("data/mock_emails.json")
        self.connected = False
        self._event_bus = event_bus
        self._mock_messages = mock_messages  # If provided, overrides file loading
        self._mock_emails: list[EmailSummary] = []

        self.credential_manager = None
        if get_credential_manager is not None:
            try:
                self.credential_manager = get_credential_manager()
            except Exception:
                self.credential_manager = None

    def connect(self) -> bool:
        """
        Connect to email service.

        Stub behavior:
- checks credentials exist (if credential manager available)
- loads mock data from file or provided mock_messages
        """
        try:
            if self.credential_manager is not None:
                try:
                    email_creds = self.credential_manager.get_credential("email")
                    if not email_creds:
                        logger.warning("No email credentials configured (continuing in stub mode)")
                except Exception as e:
                    logger.warning("Credential manager error (continuing in stub mode): %s", e)

            self._load_mock_data()
            self.connected = True
            logger.info("Email service connected (stub mode)")
            return True
        except Exception as e:
            logger.error("Failed to connect email service: %s", e, exc_info=True)
            self.connected = False
            return False

    def _publish(self, topic: str, payload: dict[str, Any]) -> None:
        if self._event_bus is None:
            return
        try:
            self._event_bus.publish(topic, payload)  # type: ignore[attr-defined]
        except Exception as e:
            logger.debug("EventBus publish failed for %s: %s", topic, e)

    def _load_mock_data(self) -> None:
        """Load mock email data from file or legacy mock messages."""
        if self._mock_messages is not None:
            self._mock_emails = [self._email_summary_from_message(m) for m in self._mock_messages]
            return

        if not self.mock_data_file.exists():
            logger.warning("No mock email data at %s", self.mock_data_file)
            self._mock_emails = []
            return

        try:
            raw = self.mock_data_file.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as e:
            logger.error("Failed to read mock email data: %s", e, exc_info=True)
            self._mock_emails = []
            return

        # Accept either:
        # 1) list[dict]
        # 2) {"messages": list[dict]}
        items: list[dict[str, Any]]
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and isinstance(data.get("messages"), list):
            items = [x for x in data["messages"] if isinstance(x, dict)]
        else:
            logger.warning("Mock email data format not recognized, expected list or {messages: [...]}")
            self._mock_emails = []
            return

        parsed: list[EmailSummary] = []
        for item in items:
            try:
                parsed.append(self._email_summary_from_dict(item))
            except Exception as e:
                logger.warning("Skipping invalid mock email record: %s", e)

        self._mock_emails = parsed
        logger.info("Loaded %d mock emails", len(self._mock_emails))

    def _email_summary_from_message(self, msg: EmailMessage) -> EmailSummary:
        return EmailSummary(
            id=msg.message_id,
            from_addr=msg.sender,
            subject=msg.subject,
            snippet=(msg.body or "")[:200],
            received_at=msg.received_at,
            labels=["unread"],
            importance_score=0.5,
        )

    def _email_summary_from_dict(self, d: dict[str, Any]) -> EmailSummary:
        # Support both schema variants:
        # - legacy: message_id, sender, subject, body, received_at
        # - newer: id, from_addr, subject, snippet, received_at, labels, importance_score
        email_id = str(d.get("id") or d.get("message_id") or "")
        if not email_id:
            raise ValueError("missing id/message_id")

        from_addr = str(d.get("from_addr") or d.get("sender") or "unknown@example.com")
        subject = str(d.get("subject") or "")
        body = d.get("body")
        snippet = d.get("snippet")

        if snippet is None and body is not None:
            snippet = str(body)[:200]
        if snippet is None:
            snippet = ""

        received_at_raw = d.get("received_at")
        received_at = self._parse_datetime(received_at_raw) or datetime.now(timezone.utc)

        labels = d.get("labels")
        if not isinstance(labels, list):
            labels = ["unread"]

        importance_score = d.get("importance_score")
        try:
            importance_score_f = float(importance_score) if importance_score is not None else 0.5
        except Exception:
            importance_score_f = 0.5

        category = d.get("category")
        category_str = str(category) if category is not None else None

        return EmailSummary(
            id=email_id,
            from_addr=from_addr,
            subject=subject,
            snippet=str(snippet),
            received_at=received_at,
            labels=[str(x) for x in labels],
            importance_score=importance_score_f,
            category=category_str,
        )

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
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

    def fetch_unread(self, limit: int = 10) -> list[EmailSummary]:
        """
        Fetch unread email summaries.

        Args:
            limit: Maximum number of emails to return

        Returns:
            List of EmailSummary objects
        """
        if not self.connected:
            if not self.connect():
                logger.warning("Email service not connected")
                return []

        unread = [email for email in self._mock_emails if "unread" in (email.labels or [])]
        result = unread[: max(0, int(limit))]
        self._publish("email.unread", {"count": len(result), "messages": [self._summary_dict(e) for e in result]})
        return result

    def triage_unread(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Legacy-friendly triage output used by older CLI command.

        Returns list of dicts:
            {
              message_id, sender, subject, received_at (datetime),
              category, summary
            }
        """
        unread = self.fetch_unread(limit=limit)
        triaged: list[dict[str, Any]] = []

        for email in unread:
            category = self.categorize(email)
            summary = self._triage_summary(email)
            triaged.append(
                {
                    "message_id": email.id,
                    "sender": email.from_addr,
                    "subject": email.subject,
                    "received_at": email.received_at,
                    "category": category,
                    "summary": summary,
                }
            )

        self._publish(
            "email.triaged",
            {"count": len(triaged), "triaged": [{"message_id": t["message_id"], "category": t["category"]} for t in triaged]},
        )
        return triaged

    def mark_as_read(self, email_id: str) -> bool:
        """Mark an email as read (stub implementation)."""
        if not self.connected:
            logger.warning("Email service not connected")
            return False

        for email in self._mock_emails:
            if email.id == email_id:
                if email.labels and "unread" in email.labels:
                    email.labels.remove("unread")
                logger.info("Marked email %s as read", email_id)
                self._publish("email.read", {"id": email_id})
                return True

        logger.warning("Email not found: %s", email_id)
        return False

    def categorize(self, email: EmailSummary) -> str:
        """
        Categorize an email based on heuristics.

        Returns:
            Category string: important, promo, social, newsletter, finance, calendar, general
        """
        subject_lower = (email.subject or "").lower()
        from_lower = (email.from_addr or "").lower()
        snippet_lower = (email.snippet or "").lower()

        # Finance indicators
        finance_keywords = ["invoice", "payment", "receipt", "billing", "charged", "refund"]
        if any(kw in subject_lower or kw in snippet_lower for kw in finance_keywords):
            return "finance"

        # Calendar indicators
        calendar_keywords = ["meeting", "schedule", "invite", "calendar", "appointment"]
        if any(kw in subject_lower or kw in snippet_lower for kw in calendar_keywords):
            return "calendar"

        # Promo indicators
        promo_keywords = ["sale", "discount", "offer", "deal", "promotion", "coupon", "free shipping"]
        if any(kw in subject_lower or kw in snippet_lower for kw in promo_keywords):
            return "promo"

        # Social indicators
        social_keywords = ["liked your", "commented on", "mentioned you", "friend request", "connection"]
        social_domains = ["facebook.com", "twitter.com", "linkedin.com", "instagram.com"]
        if any(kw in subject_lower or kw in snippet_lower for kw in social_keywords):
            return "social"
        if any(domain in from_lower for domain in social_domains):
            return "social"

        # Newsletter indicators
        if "unsubscribe" in snippet_lower or "newsletter" in subject_lower:
            return "newsletter"

        # Important indicators
        important_keywords = ["urgent", "important", "asap", "action required", "deadline"]
        if any(kw in subject_lower for kw in important_keywords):
            return "important"
        if getattr(email, "importance_score", 0.5) >= 0.8:
            return "important"

        return "general"

    def summarize(self, email_id: str) -> str:
        """
        Get a simple summary of an email (stub implementation).
        Currently returns a formatted snippet.
        """
        if not self.connected:
            return "Email service not connected"

        for email in self._mock_emails:
            if email.id == email_id:
                return f"From: {email.from_addr}\nSubject: {email.subject}\n\n{email.snippet}"

        return f"Email not found: {email_id}"

    def _triage_summary(self, email: EmailSummary) -> str:
        return f"{email.subject} (from {email.from_addr})"

    def _summary_dict(self, email: EmailSummary) -> dict[str, Any]:
        return {
            "id": email.id,
            "from_addr": email.from_addr,
            "subject": email.subject,
            "received_at": email.received_at.isoformat() if isinstance(email.received_at, datetime) else str(email.received_at),
            "labels": list(email.labels or []),
            "importance_score": float(getattr(email, "importance_score", 0.5)),
            "category": getattr(email, "category", None),
        }

    def get_all_emails(self) -> list[EmailSummary]:
        """Get all emails (stub/testing)."""
        return self._mock_emails.copy()


# Global email service instance
_email_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """Get the global email service instance."""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service


def set_email_service(service: EmailService) -> None:
    """Set the global email service instance (for testing)."""
    global _email_service
    _email_service = service


__all__ = [
    "EmailSummary",
    "EmailMessage",
    "EmailService",
    "get_email_service",
    "set_email_service",
]
