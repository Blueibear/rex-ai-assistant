"""Tests for US-P4-006: Policy-gated tool callables.

Covers:
- email_tool: module import, send_email callable
- sms_tool: module import, send_sms callable
- calendar_tool: module import, calendar_create callable
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# email_tool tests
# ---------------------------------------------------------------------------


class TestEmailTool:
    """Tests for rex.openclaw.tools.email_tool."""

    def test_import(self):
        from rex.openclaw.tools import email_tool  # noqa: F401

    def test_constants(self):
        from rex.openclaw.tools.email_tool import TOOL_DESCRIPTION, TOOL_NAME

        assert TOOL_NAME == "send_email"
        assert "email" in TOOL_DESCRIPTION.lower()

    def test_send_email_calls_service(self):
        """send_email delegates to EmailService.send with correct kwargs."""
        from rex.openclaw.tools.email_tool import send_email

        mock_service = MagicMock()
        mock_service.send.return_value = {"ok": True, "message_id": "msg-1", "error": None}

        with patch("rex.openclaw.tools.email_tool._get_email_service", return_value=mock_service):
            result = send_email("alice@example.com", "Hello", "Hi Alice!")

        mock_service.send.assert_called_once_with(
            to="alice@example.com", subject="Hello", body="Hi Alice!"
        )
        assert result == {"ok": True, "message_id": "msg-1", "error": None}

    def test_send_email_to_list(self):
        """send_email accepts a list of recipients."""
        from rex.openclaw.tools.email_tool import send_email

        mock_service = MagicMock()
        mock_service.send.return_value = {"ok": True, "message_id": "msg-2", "error": None}

        with patch("rex.openclaw.tools.email_tool._get_email_service", return_value=mock_service):
            result = send_email(["a@x.com", "b@x.com"], "Sub", "Body")

        mock_service.send.assert_called_once_with(
            to=["a@x.com", "b@x.com"], subject="Sub", body="Body"
        )
        assert result["ok"] is True

    def test_send_email_context_ignored(self):
        """context kwarg is accepted without error."""
        from rex.openclaw.tools.email_tool import send_email

        mock_service = MagicMock()
        mock_service.send.return_value = {"ok": True, "message_id": None, "error": None}

        with patch("rex.openclaw.tools.email_tool._get_email_service", return_value=mock_service):
            result = send_email("x@example.com", "S", "B", context={"location": "London"})

        assert result["ok"] is True


# ---------------------------------------------------------------------------
# sms_tool tests
# ---------------------------------------------------------------------------


class TestSmsTool:
    """Tests for rex.openclaw.tools.sms_tool."""

    def test_import(self):
        from rex.openclaw.tools import sms_tool  # noqa: F401

    def test_constants(self):
        from rex.openclaw.tools.sms_tool import TOOL_DESCRIPTION, TOOL_NAME

        assert TOOL_NAME == "send_sms"
        assert "sms" in TOOL_DESCRIPTION.lower()

    def test_send_sms_returns_stub_error(self):
        """send_sms stub returns ok=False with an error message (SMS backend retired)."""
        from rex.openclaw.tools.sms_tool import send_sms

        result = send_sms("+15551234567", "Hi there!")

        assert result["ok"] is False
        assert result["message_id"] is None
        assert "not available" in result["error"].lower()

    def test_send_sms_returns_dict_keys(self):
        """send_sms always returns a dict with ok, message_id, and error keys."""
        from rex.openclaw.tools.sms_tool import send_sms

        result = send_sms("+15559876543", "Test body")

        assert "ok" in result
        assert "message_id" in result
        assert "error" in result

    def test_send_sms_context_ignored(self):
        """context kwarg is accepted without error."""
        from rex.openclaw.tools.sms_tool import send_sms

        result = send_sms("+10001112222", "ctx test", context={"timezone": "UTC"})

        assert result["ok"] is False


# ---------------------------------------------------------------------------
# calendar_tool tests
# ---------------------------------------------------------------------------


class TestCalendarTool:
    """Tests for rex.openclaw.tools.calendar_tool."""

    def test_import(self):
        from rex.openclaw.tools import calendar_tool  # noqa: F401

    def test_constants(self):
        from rex.openclaw.tools.calendar_tool import TOOL_DESCRIPTION, TOOL_NAME

        assert TOOL_NAME == "calendar_create"
        assert "calendar" in TOOL_DESCRIPTION.lower()

    def _make_event(self, event_id: str = "evt-1", title: str = "Meeting") -> MagicMock:
        event = MagicMock()
        event.event_id = event_id
        event.title = title
        return event

    def test_calendar_create_calls_service(self):
        """calendar_create delegates to CalendarService.create_event."""
        from rex.openclaw.tools.calendar_tool import calendar_create

        event = self._make_event()
        mock_service = MagicMock()
        mock_service.create_event.return_value = event

        start = "2026-03-23T09:00:00"
        end = "2026-03-23T09:30:00"

        with patch(
            "rex.openclaw.tools.calendar_tool._get_calendar_service", return_value=mock_service
        ):
            result = calendar_create("Meeting", start, end)

        mock_service.create_event.assert_called_once()
        call_kwargs = mock_service.create_event.call_args
        assert call_kwargs.kwargs["title"] == "Meeting"
        assert isinstance(call_kwargs.kwargs["start_time"], datetime)
        assert isinstance(call_kwargs.kwargs["end_time"], datetime)
        assert result["ok"] is True
        assert result["event_id"] == "evt-1"
        assert result["title"] == "Meeting"

    def test_calendar_create_with_datetime_objects(self):
        """calendar_create accepts datetime objects as well as strings."""
        from rex.openclaw.tools.calendar_tool import calendar_create

        event = self._make_event("evt-2", "Standup")
        mock_service = MagicMock()
        mock_service.create_event.return_value = event

        start_dt = datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc)
        end_dt = datetime(2026, 3, 23, 9, 30, tzinfo=timezone.utc)

        with patch(
            "rex.openclaw.tools.calendar_tool._get_calendar_service", return_value=mock_service
        ):
            result = calendar_create("Standup", start_dt, end_dt)

        assert result["ok"] is True
        assert result["title"] == "Standup"

    def test_calendar_create_naive_string_gets_utc(self):
        """Naive ISO strings are treated as UTC (tzinfo added)."""
        from rex.openclaw.tools.calendar_tool import calendar_create

        event = self._make_event()
        mock_service = MagicMock()
        mock_service.create_event.return_value = event

        with patch(
            "rex.openclaw.tools.calendar_tool._get_calendar_service", return_value=mock_service
        ):
            calendar_create("X", "2026-04-01T10:00:00", "2026-04-01T11:00:00")

        start_arg = mock_service.create_event.call_args.kwargs["start_time"]
        assert start_arg.tzinfo is not None

    def test_calendar_create_optional_fields(self):
        """location and description are forwarded when supplied."""
        from rex.openclaw.tools.calendar_tool import calendar_create

        event = self._make_event()
        mock_service = MagicMock()
        mock_service.create_event.return_value = event

        with patch(
            "rex.openclaw.tools.calendar_tool._get_calendar_service", return_value=mock_service
        ):
            calendar_create(
                "Offsite",
                "2026-05-01T08:00:00",
                "2026-05-01T17:00:00",
                location="London HQ",
                description="Annual offsite",
            )

        kwargs = mock_service.create_event.call_args.kwargs
        assert kwargs["location"] == "London HQ"
        assert kwargs["description"] == "Annual offsite"
