"""Tests for the email service."""

from __future__ import annotations

from datetime import datetime, timezone

from rex.email_service import EmailMessage, EmailService
from rex.event_bus import EventBus


def test_email_triage_publishes_events():
    bus = EventBus()
    events = []

    def handler(event_type: str, payload: dict[str, object]) -> None:
        events.append((event_type, payload))

    bus.subscribe("email.triaged", handler)

    message = EmailMessage(
        message_id="email-123",
        sender="billing@example.com",
        subject="Invoice ready",
        body="Details inside",
        received_at=datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc),
    )
    service = EmailService(bus, mock_messages=[message])

    triaged = service.triage_unread()

    assert triaged[0]["category"] == "finance"
    assert events[0][0] == "email.triaged"
    assert events[0][1]["count"] == 1
