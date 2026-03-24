"""Unit tests for rex.integrations.models (US-241)."""

from __future__ import annotations

from datetime import datetime, timezone

from rex.integrations.models import CalendarEvent, EmailMessage

# ---------------------------------------------------------------------------
# EmailMessage tests
# ---------------------------------------------------------------------------


class TestEmailMessage:
    def _make(self, **kwargs: object) -> EmailMessage:
        defaults: dict[str, object] = {
            "id": "msg-1",
            "thread_id": "thread-1",
            "subject": "Hello",
            "sender": "alice@example.com",
        }
        defaults.update(kwargs)
        return EmailMessage(**defaults)  # type: ignore[arg-type]

    def test_minimal_instantiation(self) -> None:
        msg = self._make()
        assert msg.id == "msg-1"
        assert msg.thread_id == "thread-1"
        assert msg.subject == "Hello"
        assert msg.sender == "alice@example.com"

    def test_defaults(self) -> None:
        msg = self._make()
        assert msg.recipients == []
        assert msg.body_text == ""
        assert msg.body_html is None
        assert msg.labels == []
        assert msg.is_read is False
        assert msg.priority == "low"

    def test_optional_fields(self) -> None:
        msg = self._make(
            body_text="Hi there",
            body_html="<p>Hi there</p>",
            labels=["INBOX", "UNREAD"],
            is_read=True,
            priority="critical",
        )
        assert msg.body_text == "Hi there"
        assert msg.body_html == "<p>Hi there</p>"
        assert msg.labels == ["INBOX", "UNREAD"]
        assert msg.is_read is True
        assert msg.priority == "critical"

    def test_priority_values(self) -> None:
        for level in ("low", "medium", "high", "critical"):
            msg = self._make(priority=level)
            assert msg.priority == level

    def test_received_at_default_is_utc(self) -> None:
        msg = self._make()
        assert msg.received_at.tzinfo is not None

    def test_received_at_custom(self) -> None:
        dt = datetime(2025, 1, 15, 9, 0, tzinfo=timezone.utc)
        msg = self._make(received_at=dt)
        assert msg.received_at == dt

    def test_model_dump_round_trip(self) -> None:
        dt = datetime(2025, 1, 15, 9, 0, tzinfo=timezone.utc)
        msg = self._make(
            recipients=["bob@example.com"],
            body_text="content",
            received_at=dt,
            is_read=True,
            priority="high",
        )
        data = msg.model_dump()
        restored = EmailMessage(**data)
        assert restored == msg

    def test_model_dump_json_round_trip(self) -> None:
        msg = self._make(priority="medium", body_text="test")
        json_str = msg.model_dump_json()
        restored = EmailMessage.model_validate_json(json_str)
        assert restored.id == msg.id
        assert restored.priority == msg.priority


# ---------------------------------------------------------------------------
# CalendarEvent tests
# ---------------------------------------------------------------------------


class TestCalendarEvent:
    _start = datetime(2025, 6, 1, 10, 0, tzinfo=timezone.utc)
    _end = datetime(2025, 6, 1, 11, 0, tzinfo=timezone.utc)

    def _make(self, **kwargs: object) -> CalendarEvent:
        defaults: dict[str, object] = {
            "id": "evt-1",
            "title": "Team sync",
            "start": self._start,
            "end": self._end,
        }
        defaults.update(kwargs)
        return CalendarEvent(**defaults)  # type: ignore[arg-type]

    def test_minimal_instantiation(self) -> None:
        evt = self._make()
        assert evt.id == "evt-1"
        assert evt.title == "Team sync"
        assert evt.start == self._start
        assert evt.end == self._end

    def test_defaults(self) -> None:
        evt = self._make()
        assert evt.location is None
        assert evt.description is None
        assert evt.attendees == []
        assert evt.source == "rex"
        assert evt.is_all_day is False
        assert evt.recurrence is None

    def test_optional_fields(self) -> None:
        evt = self._make(
            location="Room 42",
            description="Weekly sync",
            attendees=["alice@example.com", "bob@example.com"],
            source="google",
            is_all_day=True,
            recurrence="RRULE:FREQ=WEEKLY;BYDAY=MO",
        )
        assert evt.location == "Room 42"
        assert evt.description == "Weekly sync"
        assert len(evt.attendees) == 2
        assert evt.source == "google"
        assert evt.is_all_day is True
        assert evt.recurrence == "RRULE:FREQ=WEEKLY;BYDAY=MO"

    def test_model_dump_round_trip(self) -> None:
        evt = self._make(
            location="HQ",
            attendees=["alice@example.com"],
            source="stub",
        )
        data = evt.model_dump()
        restored = CalendarEvent(**data)
        assert restored == evt

    def test_model_dump_json_round_trip(self) -> None:
        evt = self._make(description="Important meeting")
        json_str = evt.model_dump_json()
        restored = CalendarEvent.model_validate_json(json_str)
        assert restored.id == evt.id
        assert restored.description == evt.description
