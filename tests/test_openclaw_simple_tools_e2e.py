"""Tests for US-P4-005: simple tools through ToolBridge.

Verifies that after calling register_simple_tools(), the time_now and
weather_now tools return correct results when invoked through the bridge.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from rex.openclaw.tool_bridge import ToolBridge


class TestRegisterThenExerciseTimeTool:
    """register_simple_tools() then execute time_now through the bridge."""

    def setup_method(self):
        self.bridge = ToolBridge()

    def test_register_simple_tools_returns_handle_dict(self):
        handles = self.bridge.register_simple_tools()
        assert isinstance(handles, dict)
        assert "time_now" in handles
        assert "weather_now" in handles

    def test_time_now_callable_after_register(self):
        """After registration, execute_tool dispatches time_now correctly."""
        fake_result = {"local_time": "2026-03-22 14:30", "date": "2026-03-22", "timezone": "UTC"}
        self.bridge.register_simple_tools()
        with patch("rex.openclaw.tool_bridge._execute_tool", return_value=fake_result):
            result = self.bridge.execute_tool({"tool": "time_now", "args": {}}, {})
        assert result == fake_result

    def test_weather_now_callable_after_register(self):
        """After registration, execute_tool dispatches weather_now correctly."""
        fake_result = {"temperature": 12.0, "description": "clear sky"}
        self.bridge.register_simple_tools()
        with patch("rex.openclaw.tool_bridge._execute_tool", return_value=fake_result):
            result = self.bridge.execute_tool(
                {"tool": "weather_now", "args": {"location": "London"}}, {}
            )
        assert result == fake_result


class TestTimeNowThroughBridge:
    """time_now invoked via bridge.execute_tool returns expected structure."""

    def setup_method(self):
        self.bridge = ToolBridge()

    def test_returns_dict(self):
        fake = {"local_time": "2026-03-22 09:00", "date": "2026-03-22", "timezone": "Europe/London"}
        with patch("rex.openclaw.tool_bridge._execute_tool", return_value=fake):
            result = self.bridge.execute_tool({"tool": "time_now", "args": {}}, {})
        assert isinstance(result, dict)

    def test_result_contains_local_time(self):
        fake = {"local_time": "2026-03-22 09:00", "date": "2026-03-22", "timezone": "Europe/London"}
        with patch("rex.openclaw.tool_bridge._execute_tool", return_value=fake):
            result = self.bridge.execute_tool({"tool": "time_now", "args": {}}, {})
        assert "local_time" in result

    def test_result_contains_date(self):
        fake = {"local_time": "2026-03-22 09:00", "date": "2026-03-22", "timezone": "Europe/London"}
        with patch("rex.openclaw.tool_bridge._execute_tool", return_value=fake):
            result = self.bridge.execute_tool({"tool": "time_now", "args": {}}, {})
        assert "date" in result

    def test_result_contains_timezone(self):
        fake = {"local_time": "2026-03-22 09:00", "date": "2026-03-22", "timezone": "Europe/London"}
        with patch("rex.openclaw.tool_bridge._execute_tool", return_value=fake):
            result = self.bridge.execute_tool({"tool": "time_now", "args": {}}, {})
        assert "timezone" in result

    def test_location_arg_forwarded(self):
        """Bridge passes location arg through to the underlying tool call."""
        with patch("rex.openclaw.tool_bridge._execute_tool", return_value={}) as mock_exec:
            self.bridge.execute_tool({"tool": "time_now", "args": {"location": "Tokyo"}}, {})
        req = mock_exec.call_args.args[0]
        assert req["tool"] == "time_now"
        assert req["args"]["location"] == "Tokyo"

    def test_real_call_returns_dict(self):
        """Real (non-mocked) call to time_now returns a dict."""
        result = self.bridge.execute_tool({"tool": "time_now", "args": {"location": "London"}}, {})
        assert isinstance(result, dict)

    def test_real_call_result_has_expected_keys_or_error(self):
        """Real call returns either time keys or an error key — never empty."""
        result = self.bridge.execute_tool({"tool": "time_now", "args": {"location": "London"}}, {})
        has_time_keys = "local_time" in result or "time" in result
        has_error = "error" in result
        assert has_time_keys or has_error, f"Unexpected result shape: {result}"


class TestWeatherNowThroughBridge:
    """weather_now invoked via bridge.execute_tool returns expected structure."""

    def setup_method(self):
        self.bridge = ToolBridge()

    def test_returns_dict(self):
        fake = {"temperature": 15.0, "description": "sunny", "location": "Paris"}
        with patch("rex.openclaw.tool_bridge._execute_tool", return_value=fake):
            result = self.bridge.execute_tool(
                {"tool": "weather_now", "args": {"location": "Paris"}}, {}
            )
        assert isinstance(result, dict)

    def test_success_result_structure(self):
        fake = {"temperature": 15.0, "description": "sunny", "location": "Paris"}
        with patch("rex.openclaw.tool_bridge._execute_tool", return_value=fake):
            result = self.bridge.execute_tool(
                {"tool": "weather_now", "args": {"location": "Paris"}}, {}
            )
        assert result["temperature"] == 15.0
        assert result["description"] == "sunny"

    def test_location_arg_forwarded(self):
        """Bridge passes location arg through to weather_now call."""
        with patch("rex.openclaw.tool_bridge._execute_tool", return_value={}) as mock_exec:
            self.bridge.execute_tool({"tool": "weather_now", "args": {"location": "Berlin"}}, {})
        req = mock_exec.call_args.args[0]
        assert req["tool"] == "weather_now"
        assert req["args"]["location"] == "Berlin"

    def test_real_call_returns_dict(self):
        """Real call (credentials bypassed) returns a dict."""
        result = self.bridge.execute_tool(
            {"tool": "weather_now", "args": {"location": "London"}},
            {},
            skip_credential_check=True,
        )
        assert isinstance(result, dict)

    def test_real_call_without_api_key_returns_error_or_weather(self):
        """With credentials bypassed the result is weather data or an error dict."""
        result = self.bridge.execute_tool(
            {"tool": "weather_now", "args": {"location": "London"}},
            {},
            skip_credential_check=True,
        )
        has_weather_data = "temperature" in result or "temp" in result
        has_error = "error" in result
        assert has_weather_data or has_error, f"Unexpected result shape: {result}"


class TestSimpleToolsViaRouteIfToolRequest:
    """End-to-end: route_if_tool_request detects and executes simple tools."""

    def setup_method(self):
        self.bridge = ToolBridge()

    def test_time_now_tool_request_routed(self):
        """A TOOL_REQUEST line for time_now is dispatched and model re-called."""
        llm_output = 'TOOL_REQUEST: {"tool": "time_now", "args": {}}'
        fake_tool_result = {
            "local_time": "2026-03-22 14:00",
            "date": "2026-03-22",
            "timezone": "UTC",
        }
        model_fn = MagicMock(return_value="It is 2:00 PM.")

        with patch("rex.openclaw.tool_bridge._execute_tool", return_value=fake_tool_result):
            reply = self.bridge.route_if_tool_request(llm_output, {}, model_fn)

        model_fn.assert_called_once()
        assert isinstance(reply, str)

    def test_weather_now_tool_request_routed(self):
        """A TOOL_REQUEST line for weather_now is dispatched and model re-called.

        route_if_tool_request delegates to rex.openclaw.tool_executor.route_if_tool_request,
        which calls rex.openclaw.tool_executor.execute_tool internally.  Patching at the
        tool_executor level ensures the credential check is bypassed.
        """
        llm_output = 'TOOL_REQUEST: {"tool": "weather_now", "args": {"location": "London"}}'
        fake_tool_result = {"temperature": 12.0, "description": "cloudy"}
        model_fn = MagicMock(return_value="It is cloudy in London.")

        with patch("rex.openclaw.tool_executor.execute_tool", return_value=fake_tool_result):
            reply = self.bridge.route_if_tool_request(llm_output, {}, model_fn)

        model_fn.assert_called_once()
        assert isinstance(reply, str)

    def test_non_tool_text_returned_unchanged(self):
        """Plain text that is not a tool request passes through unmodified."""
        result = self.bridge.route_if_tool_request(
            "Hello, how can I help?", {}, lambda msg: "unreachable"
        )
        assert result == "Hello, how can I help?"
