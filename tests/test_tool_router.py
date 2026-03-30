"""Tests for rex.local_tool_executor - handlers and UnknownToolError."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rex.tool_catalog import EXECUTABLE_TOOLS
from rex.local_tool_executor import UnknownToolError, execute_tool


class TestUnknownToolError:
    def test_raises_for_unknown_tool(self):
        with pytest.raises(UnknownToolError) as exc_info:
            execute_tool("totally_fake_tool", {})
        assert "totally_fake_tool" in str(exc_info.value)

    def test_error_carries_tool_name(self):
        try:
            execute_tool("nonexistent", {})
        except UnknownToolError as exc:
            assert exc.tool_name == "nonexistent"

    def test_error_message_lists_catalog(self):
        try:
            execute_tool("bogus", {})
        except UnknownToolError as exc:
            for tool in EXECUTABLE_TOOLS:
                assert tool in str(exc)

    def test_unknown_tool_is_not_generic_exception(self):
        with pytest.raises(UnknownToolError):
            execute_tool("not_in_catalog", {})
        try:
            execute_tool("not_in_catalog", {})
        except (ValueError, KeyError):
            pytest.fail("Should be UnknownToolError, not ValueError/KeyError")
        except UnknownToolError:
            pass


class TestExecuteToolCatalogCoverage:
    """Every tool in EXECUTABLE_TOOLS must be callable without raising."""

    @pytest.mark.parametrize("tool_name", sorted(EXECUTABLE_TOOLS))
    def test_all_catalog_tools_return_string(self, tool_name: str):
        result = execute_tool(tool_name, {})
        assert isinstance(result, str), f"{tool_name} did not return str"
        assert len(result) > 0, f"{tool_name} returned empty string"

    @pytest.mark.parametrize("tool_name", sorted(EXECUTABLE_TOOLS))
    def test_catalog_tools_do_not_raise(self, tool_name: str):
        try:
            execute_tool(tool_name, {})
        except UnknownToolError:
            pytest.fail(f"{tool_name} is in EXECUTABLE_TOOLS but raised UnknownToolError")


class TestTimeNow:
    def test_returns_time_string(self):
        result = execute_tool("time_now", {"location": "UTC"})
        assert "UTC" in result or "time" in result.lower()

    def test_empty_args_ok(self):
        result = execute_tool("time_now", {})
        assert isinstance(result, str) and len(result) > 0

    def test_location_reflected_in_output(self):
        result = execute_tool("time_now", {"location": "New York"})
        assert "New York" in result


class TestWeatherNow:
    def test_no_api_key_returns_not_configured(self):
        """Missing openweathermap key → sentinel string (no exception)."""
        mock_cm = MagicMock()
        mock_cm.get_token.return_value = None
        with patch("rex.local_tool_executor.get_credential_manager", return_value=mock_cm):
            result = execute_tool("weather_now", {"location": "London"})
        assert result == "[integration not configured]"

    def test_with_api_key_returns_formatted_weather(self):
        """With a mocked provider response, result is a readable string."""
        mock_cm = MagicMock()
        mock_cm.get_token.return_value = "fake-owm-key"

        mock_weather_data = {
            "city": "London",
            "description": "light rain",
            "temp_f": 55.4,
            "temp_c": 13.0,
            "humidity": 80,
            "wind_mph": 12.3,
        }

        async def fake_get_weather(location: str, api_key: str):
            return mock_weather_data

        with patch("rex.local_tool_executor.get_credential_manager", return_value=mock_cm):
            with patch("rex.local_tool_executor.get_weather", side_effect=fake_get_weather):
                result = execute_tool("weather_now", {"location": "London"})

        assert "London" in result
        assert "light rain" in result
        assert "55.4" in result
        assert "80%" in result

    def test_provider_error_returns_error_string(self):
        """If the weather API returns an error dict, result is an error string."""
        mock_cm = MagicMock()
        mock_cm.get_token.return_value = "fake-owm-key"

        async def fake_get_weather(location: str, api_key: str):
            return {"error": "City not found"}

        with patch("rex.local_tool_executor.get_credential_manager", return_value=mock_cm):
            with patch("rex.local_tool_executor.get_weather", side_effect=fake_get_weather):
                result = execute_tool("weather_now", {"location": "NoSuchPlace"})

        assert "[weather error:" in result
        assert "City not found" in result

    def test_empty_location_uses_default(self):
        """Empty location arg is handled gracefully (returns sentinel or result)."""
        mock_cm = MagicMock()
        mock_cm.get_token.return_value = None  # no key → not configured
        with patch("rex.local_tool_executor.get_credential_manager", return_value=mock_cm):
            result = execute_tool("weather_now", {})
        assert isinstance(result, str) and len(result) > 0


class TestWebSearch:
    def test_no_results_returns_not_configured(self):
        """If search_web returns None, return sentinel string."""
        with patch("plugins.web_search.search_web", return_value=None):
            result = execute_tool("web_search", {"query": "test query"})
        assert result == "[integration not configured]"

    def test_with_results_returns_summary(self):
        """With a mocked provider, result contains the returned text."""
        mock_result = "Example Title - https://example.com\nA sample snippet about the topic."
        with patch("plugins.web_search.search_web", return_value=mock_result):
            result = execute_tool("web_search", {"query": "test query"})
        assert "Example Title" in result
        assert "example.com" in result

    def test_missing_query_returns_error_message(self):
        """Missing query argument returns a descriptive error string."""
        result = execute_tool("web_search", {})
        assert "query" in result.lower()

    def test_provider_exception_returns_not_configured(self):
        """If the search provider raises, return sentinel (no exception propagation)."""
        with patch("plugins.web_search.search_web", side_effect=RuntimeError("network error")):
            result = execute_tool("web_search", {"query": "something"})
        assert result == "[integration not configured]"


class TestSendEmail:
    def test_returns_email_sent_on_success(self):
        """EmailService.send() returning ok=True → 'Email sent'."""
        mock_svc = MagicMock()
        mock_svc.send.return_value = {"ok": True, "message_id": "msg-1", "error": None}
        with patch("rex.local_tool_executor.EmailService", return_value=mock_svc):
            result = execute_tool(
                "send_email",
                {"to": "alice@example.com", "subject": "Hi", "body": "Hello"},
            )
        assert result == "Email sent"
        mock_svc.send.assert_called_once_with(to="alice@example.com", subject="Hi", body="Hello")

    def test_returns_error_string_on_failure(self):
        """EmailService.send() returning ok=False → error string."""
        mock_svc = MagicMock()
        mock_svc.send.return_value = {"ok": False, "message_id": None, "error": "Auth failed"}
        with patch("rex.local_tool_executor.EmailService", return_value=mock_svc):
            result = execute_tool(
                "send_email",
                {"to": "alice@example.com", "subject": "Hi", "body": "Hello"},
            )
        assert "[send_email error:" in result
        assert "Auth failed" in result

    def test_missing_to_returns_error(self):
        result = execute_tool("send_email", {"subject": "Hi", "body": "Hello"})
        assert "[send_email error:" in result
        assert "to" in result.lower()

    def test_exception_degraded_gracefully(self):
        """If EmailService raises, return error string (no exception propagation)."""
        with patch("rex.local_tool_executor.EmailService", side_effect=RuntimeError("no connection")):
            result = execute_tool("send_email", {"to": "a@b.com", "subject": "s", "body": "b"})
        assert "[send_email error:" in result
        assert isinstance(result, str)

    def test_default_subject_used_when_missing(self):
        mock_svc = MagicMock()
        mock_svc.send.return_value = {"ok": True, "message_id": None, "error": None}
        with patch("rex.local_tool_executor.EmailService", return_value=mock_svc):
            execute_tool("send_email", {"to": "a@b.com", "body": "b"})
        call_kwargs = mock_svc.send.call_args.kwargs
        assert call_kwargs["subject"] == "(no subject)"


class TestCalendarCreateEvent:
    def test_returns_confirmation_with_title(self):
        """CalendarService.create_event() called; returns confirmation string."""
        from datetime import datetime, timezone

        fake_event = MagicMock()
        fake_event.title = "Team Meeting"
        fake_event.start_time = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
        fake_event.end_time = datetime(2026, 4, 1, 11, 0, tzinfo=timezone.utc)

        mock_svc = MagicMock()
        mock_svc.create_event.return_value = fake_event
        with patch("rex.local_tool_executor.CalendarService", return_value=mock_svc):
            result = execute_tool(
                "calendar_create_event",
                {
                    "title": "Team Meeting",
                    "start": "2026-04-01T10:00:00",
                    "end": "2026-04-01T11:00:00",
                },
            )
        assert "Calendar event created" in result
        assert "Team Meeting" in result

    def test_missing_times_uses_defaults(self):
        """Missing start/end args result in sensible defaults (no exception)."""
        fake_event = MagicMock()
        fake_event.title = "Standup"
        from datetime import datetime, timezone

        fake_event.start_time = datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc)
        fake_event.end_time = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)

        mock_svc = MagicMock()
        mock_svc.create_event.return_value = fake_event
        with patch("rex.local_tool_executor.CalendarService", return_value=mock_svc):
            result = execute_tool("calendar_create_event", {"title": "Standup"})
        assert "Calendar event created" in result

    def test_exception_degraded_gracefully(self):
        with patch("rex.local_tool_executor.CalendarService", side_effect=RuntimeError("db error")):
            result = execute_tool(
                "calendar_create_event",
                {"title": "Broken", "start": "2026-04-01T10:00:00"},
            )
        assert "[calendar error:" in result
        assert isinstance(result, str)

    def test_summary_key_also_accepted_as_title(self):
        """'summary' key (from planner) is accepted as the event title."""
        fake_event = MagicMock()
        fake_event.title = "Weekly Sync"
        from datetime import datetime, timezone

        fake_event.start_time = datetime(2026, 4, 2, 14, 0, tzinfo=timezone.utc)
        fake_event.end_time = datetime(2026, 4, 2, 15, 0, tzinfo=timezone.utc)

        mock_svc = MagicMock()
        mock_svc.create_event.return_value = fake_event
        with patch("rex.local_tool_executor.CalendarService", return_value=mock_svc):
            result = execute_tool("calendar_create_event", {"summary": "Weekly Sync"})
        assert "Calendar event created" in result


class TestNoLongerStubs:
    """Verify weather_now and web_search are no longer returning the old stub sentinel
    when a mock API key IS present (i.e., the real handler code runs)."""

    def test_weather_now_with_key_does_not_return_stub_sentinel(self):
        """With a key present, weather_now must invoke the provider (not return stub)."""
        mock_cm = MagicMock()
        mock_cm.get_token.return_value = "test-key"

        async def fake_get_weather(location: str, api_key: str):
            return {
                "city": "Paris",
                "description": "sunny",
                "temp_f": 75.0,
                "temp_c": 23.9,
                "humidity": 50,
                "wind_mph": 5.0,
            }

        with patch("rex.local_tool_executor.get_credential_manager", return_value=mock_cm):
            with patch("rex.local_tool_executor.get_weather", side_effect=fake_get_weather):
                result = execute_tool("weather_now", {"location": "Paris"})

        # Should be a real weather string, NOT the stub sentinel
        assert result != "[integration not configured]"
        assert "Paris" in result

    def test_web_search_with_result_does_not_return_stub_sentinel(self):
        """With a successful search, web_search must return the result text."""
        with patch(
            "plugins.web_search.search_web",
            return_value="Top result - https://example.com\nSnippet here",
        ):
            result = execute_tool("web_search", {"query": "test"})

        assert result != "[integration not configured]"
        assert "example.com" in result
