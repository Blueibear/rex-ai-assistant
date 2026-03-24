"""Tests for US-P4-008: Register HA tools batch.

Covers:
- ha_tool: module import, ha_call_service callable, register stub
- ToolBridge.register_ha_tools: returns expected keys, calls register fn
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# ha_tool tests
# ---------------------------------------------------------------------------


class TestHaTool:
    """Tests for rex.openclaw.tools.ha_tool."""

    def test_import(self):
        from rex.openclaw.tools import ha_tool  # noqa: F401

    def test_constants(self):
        from rex.openclaw.tools.ha_tool import TOOL_DESCRIPTION, TOOL_NAME

        assert TOOL_NAME == "home_assistant_call_service"
        assert "home assistant" in TOOL_DESCRIPTION.lower()

    def test_ha_call_service_calls_bridge(self):
        """ha_call_service constructs IntentMatch and calls _execute_intent."""
        from rex.openclaw.tools.ha_tool import ha_call_service

        mock_bridge = MagicMock()
        mock_bridge.enabled = True
        mock_bridge._execute_intent.return_value = (True, "Turn on light.living_room.")

        with patch("rex.openclaw.tools.ha_tool._get_ha_bridge", return_value=mock_bridge):
            result = ha_call_service("light", "turn_on", "light.living_room")

        mock_bridge._execute_intent.assert_called_once()
        intent_arg = mock_bridge._execute_intent.call_args.args[0]
        assert intent_arg.domain == "light"
        assert intent_arg.service == "turn_on"
        assert intent_arg.entity_id == "light.living_room"
        assert result == {
            "success": True,
            "message": "Turn on light.living_room.",
            "entity_id": "light.living_room",
        }

    def test_ha_call_service_with_data(self):
        """Extra data dict is merged into intent_data alongside entity_id."""
        from rex.openclaw.tools.ha_tool import ha_call_service

        mock_bridge = MagicMock()
        mock_bridge.enabled = True
        mock_bridge._execute_intent.return_value = (True, "Done.")

        with patch("rex.openclaw.tools.ha_tool._get_ha_bridge", return_value=mock_bridge):
            ha_call_service(
                "light",
                "turn_on",
                "light.bedroom",
                data={"brightness_pct": 50},
            )

        intent_arg = mock_bridge._execute_intent.call_args.args[0]
        assert intent_arg.data["entity_id"] == "light.bedroom"
        assert intent_arg.data["brightness_pct"] == 50

    def test_ha_call_service_not_enabled(self):
        """Returns error dict when HA bridge is not configured."""
        from rex.openclaw.tools.ha_tool import ha_call_service

        mock_bridge = MagicMock()
        mock_bridge.enabled = False

        with patch("rex.openclaw.tools.ha_tool._get_ha_bridge", return_value=mock_bridge):
            result = ha_call_service("switch", "turn_off", "switch.garage")

        assert result["success"] is False
        assert "not configured" in result["message"]
        assert result["entity_id"] == "switch.garage"
        mock_bridge._execute_intent.assert_not_called()

    def test_ha_call_service_context_ignored(self):
        """context kwarg is accepted without error."""
        from rex.openclaw.tools.ha_tool import ha_call_service

        mock_bridge = MagicMock()
        mock_bridge.enabled = True
        mock_bridge._execute_intent.return_value = (True, "Done.")

        with patch("rex.openclaw.tools.ha_tool._get_ha_bridge", return_value=mock_bridge):
            result = ha_call_service(
                "light",
                "turn_off",
                "light.office",
                context={"user": "james"},
            )

        assert result["success"] is True

    def test_register_returns_none_without_openclaw(self):
        """register() returns None when openclaw gateway not configured."""
        from unittest.mock import patch

        from rex.openclaw.tools.ha_tool import register

        with patch("rex.openclaw.http_client.get_openclaw_client", return_value=None):
            assert register() is None
            assert register(agent=object()) is None


# ---------------------------------------------------------------------------
# ToolBridge.register_ha_tools tests
# ---------------------------------------------------------------------------


class TestRegisterHaTools:
    """Tests for ToolBridge.register_ha_tools."""

    def test_returns_dict(self):
        from rex.openclaw.tool_bridge import ToolBridge

        bridge = ToolBridge()
        result = bridge.register_ha_tools()
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        from rex.openclaw.tool_bridge import ToolBridge

        bridge = ToolBridge()
        result = bridge.register_ha_tools()
        assert "home_assistant_call_service" in result

    def test_no_extra_keys(self):
        from rex.openclaw.tool_bridge import ToolBridge

        bridge = ToolBridge()
        result = bridge.register_ha_tools()
        assert set(result.keys()) == {"home_assistant_call_service"}

    def test_calls_register_fn(self):
        """register_ha_tools calls the individual register function."""
        from rex.openclaw.tool_bridge import ToolBridge

        sentinel = object()

        with patch(
            "rex.openclaw.tool_bridge._register_ha_call_service", return_value=sentinel
        ) as mock_fn:
            bridge = ToolBridge()
            result = bridge.register_ha_tools()

        mock_fn.assert_called_once_with(agent=None)
        assert result["home_assistant_call_service"] is sentinel

    def test_forwards_agent_arg(self):
        """agent kwarg is forwarded to the register function."""
        from rex.openclaw.tool_bridge import ToolBridge

        fake_agent = object()

        with patch(
            "rex.openclaw.tool_bridge._register_ha_call_service", return_value=None
        ) as mock_fn:
            ToolBridge().register_ha_tools(agent=fake_agent)

        mock_fn.assert_called_once_with(agent=fake_agent)

    def test_returns_none_values_without_openclaw(self):
        """Without openclaw gateway configured, all values in the dict are None."""
        from unittest.mock import patch

        from rex.openclaw.tool_bridge import ToolBridge

        with patch("rex.openclaw.http_client.get_openclaw_client", return_value=None):
            result = ToolBridge().register_ha_tools()
            assert all(v is None for v in result.values())
