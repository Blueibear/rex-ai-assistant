"""Tests for US-P4-006: Register policy-gated tools batch.

Covers:
- email_tool: module import, send_email callable, register stub
- sms_tool: module import, send_sms callable, register stub
- calendar_tool: module import, calendar_create callable, register stub
- ToolBridge.register_policy_gated_tools: returns expected keys, calls each register fn
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

    def test_register_returns_none_without_openclaw(self):
        """register() returns None when openclaw is not installed."""
        from rex.openclaw.tools.email_tool import OPENCLAW_AVAILABLE, register

        if not OPENCLAW_AVAILABLE:
            assert register() is None
            assert register(agent=object()) is None


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

    def test_register_returns_none_without_openclaw(self):
        """register() returns None when openclaw is not installed."""
        from rex.openclaw.tools.sms_tool import OPENCLAW_AVAILABLE, register

        if not OPENCLAW_AVAILABLE:
            assert register() is None
            assert register(agent=object()) is None


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

    def test_register_returns_none_without_openclaw(self):
        """register() returns None when openclaw is not installed."""
        from rex.openclaw.tools.calendar_tool import OPENCLAW_AVAILABLE, register

        if not OPENCLAW_AVAILABLE:
            assert register() is None
            assert register(agent=object()) is None


# ---------------------------------------------------------------------------
# ToolBridge.register_policy_gated_tools tests
# ---------------------------------------------------------------------------


class TestRegisterPolicyGatedTools:
    """Tests for ToolBridge.register_policy_gated_tools."""

    def test_returns_dict(self):
        from rex.openclaw.tool_bridge import ToolBridge

        bridge = ToolBridge()
        result = bridge.register_policy_gated_tools()
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        from rex.openclaw.tool_bridge import ToolBridge

        bridge = ToolBridge()
        result = bridge.register_policy_gated_tools()
        assert "send_email" in result
        assert "send_sms" in result
        assert "calendar_create" in result

    def test_no_extra_keys(self):
        from rex.openclaw.tool_bridge import ToolBridge

        bridge = ToolBridge()
        result = bridge.register_policy_gated_tools()
        assert set(result.keys()) == {"send_email", "send_sms", "calendar_create"}

    def test_calls_each_register_fn(self):
        """register_policy_gated_tools calls the individual register functions."""
        from rex.openclaw.tool_bridge import ToolBridge

        sentinel_email = object()
        sentinel_sms = object()
        sentinel_cal = object()

        with (
            patch(
                "rex.openclaw.tool_bridge._register_send_email", return_value=sentinel_email
            ) as mock_email,
            patch(
                "rex.openclaw.tool_bridge._register_send_sms", return_value=sentinel_sms
            ) as mock_sms,
            patch(
                "rex.openclaw.tool_bridge._register_calendar_create", return_value=sentinel_cal
            ) as mock_cal,
        ):
            bridge = ToolBridge()
            result = bridge.register_policy_gated_tools()

        mock_email.assert_called_once_with(agent=None)
        mock_sms.assert_called_once_with(agent=None)
        mock_cal.assert_called_once_with(agent=None)

        assert result["send_email"] is sentinel_email
        assert result["send_sms"] is sentinel_sms
        assert result["calendar_create"] is sentinel_cal

    def test_forwards_agent_arg(self):
        """agent kwarg is forwarded to every register function."""
        from rex.openclaw.tool_bridge import ToolBridge

        fake_agent = object()

        with (
            patch("rex.openclaw.tool_bridge._register_send_email", return_value=None) as me,
            patch("rex.openclaw.tool_bridge._register_send_sms", return_value=None) as ms,
            patch("rex.openclaw.tool_bridge._register_calendar_create", return_value=None) as mc,
        ):
            ToolBridge().register_policy_gated_tools(agent=fake_agent)

        me.assert_called_once_with(agent=fake_agent)
        ms.assert_called_once_with(agent=fake_agent)
        mc.assert_called_once_with(agent=fake_agent)

    def test_no_agent_uses_none_default(self):
        """register_policy_gated_tools() with no args passes agent=None."""
        from rex.openclaw.tool_bridge import ToolBridge

        with (
            patch("rex.openclaw.tool_bridge._register_send_email", return_value=None) as me,
            patch("rex.openclaw.tool_bridge._register_send_sms", return_value=None) as ms,
            patch("rex.openclaw.tool_bridge._register_calendar_create", return_value=None) as mc,
        ):
            ToolBridge().register_policy_gated_tools()

        me.assert_called_once_with(agent=None)
        ms.assert_called_once_with(agent=None)
        mc.assert_called_once_with(agent=None)

    def test_returns_none_values_without_openclaw(self):
        """Without openclaw, register() returns None for all tools."""
        from rex.openclaw.tool_bridge import OPENCLAW_AVAILABLE, ToolBridge

        if not OPENCLAW_AVAILABLE:
            bridge = ToolBridge()
            result = bridge.register_policy_gated_tools()
            assert result["send_email"] is None
            assert result["send_sms"] is None
            assert result["calendar_create"] is None
