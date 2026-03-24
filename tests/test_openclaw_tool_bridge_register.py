"""Tests for ToolBridge.register_simple_tools — US-P4-004."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rex.openclaw.tool_bridge import ToolBridge


class TestRegisterSimpleTools:
    """ToolBridge.register_simple_tools() registers the read-only tool batch."""

    def setup_method(self):
        self.bridge = ToolBridge()

    def test_returns_dict(self):
        result = self.bridge.register_simple_tools()
        assert isinstance(result, dict)

    def test_dict_has_time_now_key(self):
        result = self.bridge.register_simple_tools()
        assert "time_now" in result

    def test_dict_has_weather_now_key(self):
        result = self.bridge.register_simple_tools()
        assert "weather_now" in result

    def test_no_extra_keys(self):
        result = self.bridge.register_simple_tools()
        assert set(result.keys()) == {"time_now", "weather_now"}

    def test_calls_time_now_register(self):
        with patch("rex.openclaw.tool_bridge._register_time_now", return_value=None) as mock_time:
            patch("rex.openclaw.tool_bridge._register_weather_now", return_value=None).start()
            self.bridge.register_simple_tools()
            mock_time.assert_called_once()

    def test_calls_weather_now_register(self):
        with patch(
            "rex.openclaw.tool_bridge._register_weather_now", return_value=None
        ) as mock_weather:
            patch("rex.openclaw.tool_bridge._register_time_now", return_value=None).start()
            self.bridge.register_simple_tools()
            mock_weather.assert_called_once()

    def test_agent_forwarded_to_time_now(self):
        agent = MagicMock()
        with (
            patch("rex.openclaw.tool_bridge._register_time_now", return_value=None) as mock_time,
            patch("rex.openclaw.tool_bridge._register_weather_now", return_value=None),
        ):
            self.bridge.register_simple_tools(agent=agent)
            mock_time.assert_called_once_with(agent=agent)

    def test_agent_forwarded_to_weather_now(self):
        agent = MagicMock()
        with (
            patch("rex.openclaw.tool_bridge._register_time_now", return_value=None),
            patch(
                "rex.openclaw.tool_bridge._register_weather_now", return_value=None
            ) as mock_weather,
        ):
            self.bridge.register_simple_tools(agent=agent)
            mock_weather.assert_called_once_with(agent=agent)

    def test_returns_none_values_without_openclaw(self):
        from unittest.mock import patch

        with patch("rex.openclaw.http_client.get_openclaw_client", return_value=None):
            result = self.bridge.register_simple_tools()
            assert result["time_now"] is None
            assert result["weather_now"] is None

    def test_registration_handles_stored(self):
        fake_handle_time = object()
        fake_handle_weather = object()
        with (
            patch(
                "rex.openclaw.tool_bridge._register_time_now",
                return_value=fake_handle_time,
            ),
            patch(
                "rex.openclaw.tool_bridge._register_weather_now",
                return_value=fake_handle_weather,
            ),
        ):
            result = self.bridge.register_simple_tools()
            assert result["time_now"] is fake_handle_time
            assert result["weather_now"] is fake_handle_weather

    def test_no_agent_uses_none_default(self):
        with (
            patch("rex.openclaw.tool_bridge._register_time_now", return_value=None) as mock_time,
            patch(
                "rex.openclaw.tool_bridge._register_weather_now", return_value=None
            ) as mock_weather,
        ):
            self.bridge.register_simple_tools()
            mock_time.assert_called_once_with(agent=None)
            mock_weather.assert_called_once_with(agent=None)
