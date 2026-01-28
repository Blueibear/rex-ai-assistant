"""Email triage service using mock data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rex.event_bus import EventBus


@dataclass(frozen=True)
class EmailMessage:
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
        }


class EmailService:
    """Read-only email triage service backed by mock data."""

    def __init__(
        self,
        event_bus: EventBus,
        *,
        mock_data_path: Path | str | None = None,
        mock_messages: list[EmailMessage] | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._mock_data_path = Path(mock_data_path) if mock_data_path else None
        self._mock_messages = mock_messages

    def fetch_unread(self) -> list[EmailMessage]:
        if self._mock_messages is not None:
            return list(self._mock_messages)
        if self._mock_data_path and self._mock_data_path.exists():
            return self._load_mock_messages(self._mock_data_path)
        return self._default_mock_messages()

    def triage_unread(self) -> list[dict[str, Any]]:
        messages = self.fetch_unread()
        self._event_bus.publish(
            "email.unread",
            {"count": len(messages), "messages": [m.to_summary() for m in messages]},
        )
        triaged = []
        for message in messages:
            category = self._categorize(message)
            summary = self._summarize(message)
            triaged.append(
                {
                    "message_id": message.message_id,
                    "sender": message.sender,
                    "subject": message.subject,
                    "received_at": message.received_at,
                    "category": category,
                    "summary": summary,
                }
            )
        self._event_bus.publish(
            "email.triaged",
            {"count": len(triaged), "triaged": self._summaries(triaged)},
        )
        return triaged

    def _categorize(self, message: EmailMessage) -> str:
        subject_lower = message.subject.lower()
        if "invoice" in subject_lower or "payment" in subject_lower:
            return "finance"
        if "meeting" in subject_lower or "schedule" in subject_lower:
            return "calendar"
        if "newsletter" in subject_lower:
            return "newsletter"
        return "general"

    def _summarize(self, message: EmailMessage) -> str:
        return f"{message.subject} (from {message.sender})"

    def _summaries(self, triaged: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "message_id": item["message_id"],
                "category": item["category"],
                "summary": item["summary"],
            }
            for item in triaged
        ]

    def _load_mock_messages(self, path: Path) -> list[EmailMessage]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        messages = []
        for item in payload.get("messages", []):
            received_at = datetime.fromisoformat(item["received_at"]).astimezone(
                timezone.utc
            )
            messages.append(
                EmailMessage(
                    message_id=item["message_id"],
                    sender=item["sender"],
                    subject=item["subject"],
                    body=item.get("body", ""),
                    received_at=received_at,
                )
            )
        return messages

    def _default_mock_messages(self) -> list[EmailMessage]:
        now = datetime.now(timezone.utc)
        return [
            EmailMessage(
                message_id="email-001",
                sender="billing@example.com",
                subject="Invoice for March services",
                body="Your invoice is attached.",
                received_at=now,
            ),
            EmailMessage(
                message_id="email-002",
                sender="team@example.com",
                subject="Schedule meeting for Q2 planning",
                body="Please review proposed times.",
                received_at=now,
            ),
        ]


__all__ = ["EmailMessage", "EmailService"]
