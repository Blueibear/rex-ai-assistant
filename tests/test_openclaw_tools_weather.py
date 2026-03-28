"""Tests for rex.openclaw.tools.weather_tool — US-P2-007."""

from __future__ import annotations

from unittest.mock import patch


class TestWeatherTool:
    def test_weather_now_returns_dict(self):
        """weather_now returns a dict (success or error)."""
        from rex.openclaw.tools.weather_tool import weather_now

        fake_result = {"temperature": 12.3, "description": "cloudy", "location": "London"}
        with patch("rex.openclaw.tools.weather_tool.execute_tool", return_value=fake_result):
            result = weather_now("London")

        assert result == fake_result

    def test_weather_now_passes_location_in_args(self):
        """weather_now forwards location to execute_tool args."""
        from rex.openclaw.tools.weather_tool import weather_now

        with patch("rex.openclaw.tools.weather_tool.execute_tool", return_value={}) as mock_exec:
            weather_now("Edinburgh, Scotland")

        call_args = mock_exec.call_args
        assert call_args.args[0]["tool"] == "weather_now"
        assert call_args.args[0]["args"]["location"] == "Edinburgh, Scotland"

    def test_weather_now_no_location_omits_from_args(self):
        """When location is None, args dict has no 'location' key."""
        from rex.openclaw.tools.weather_tool import weather_now

        with patch("rex.openclaw.tools.weather_tool.execute_tool", return_value={}) as mock_exec:
            weather_now()

        call_args = mock_exec.call_args
        assert "location" not in call_args.args[0]["args"]

    def test_weather_now_passes_context(self):
        """Context dict is forwarded as default_context to execute_tool."""
        from rex.openclaw.tools.weather_tool import weather_now

        ctx = {"location": "Paris", "timezone": "Europe/Paris"}
        with patch("rex.openclaw.tools.weather_tool.execute_tool", return_value={}) as mock_exec:
            weather_now(context=ctx)

        call_args = mock_exec.call_args
        assert call_args.args[1] == ctx

    def test_weather_now_skips_policy_and_audit(self):
        """weather_now disables policy/credential/audit checks."""
        from rex.openclaw.tools.weather_tool import weather_now

        with patch("rex.openclaw.tools.weather_tool.execute_tool", return_value={}) as mock_exec:
            weather_now("Tokyo")

        call_kwargs = mock_exec.call_args.kwargs
        assert call_kwargs.get("skip_policy_check") is True
        assert call_kwargs.get("skip_credential_check") is True
        assert call_kwargs.get("skip_audit_log") is True

    def test_weather_now_error_when_no_api_key(self):
        """Without API key, weather_now returns an error dict (not raises)."""
        import os

        from rex.openclaw.tools.weather_tool import weather_now

        # Don't mock — let it hit the real tool_executor which requires an API key
        original = os.environ.pop("OPENWEATHERMAP_API_KEY", None)
        try:
            result = weather_now("London")
        finally:
            if original is not None:
                os.environ["OPENWEATHERMAP_API_KEY"] = original

        assert isinstance(result, dict)
        # When key is absent, tool_executor returns {"error": {...}}
        assert "error" in result

    def test_tool_name_constant(self):
        from rex.openclaw.tools import weather_tool

        assert weather_tool.TOOL_NAME == "weather_now"

    def test_tool_description_is_string(self):
        from rex.openclaw.tools import weather_tool

        assert isinstance(weather_tool.TOOL_DESCRIPTION, str)
        assert len(weather_tool.TOOL_DESCRIPTION) > 0

    def test_weather_tool_importable(self):
        import rex.openclaw.tools.weather_tool  # noqa: F401
