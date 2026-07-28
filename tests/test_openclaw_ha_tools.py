"""Tests for US-P4-008: HA tool callable.

Covers:
- ha_tool: module import, ha_call_service callable
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
        """ha_call_service constructs a typed mutation and calls the policy service."""
        from rex.ha.mutation_service import HAMutationResult, HAOutcome, HARisk
        from rex.openclaw.tools.ha_tool import ha_call_service

        mock_bridge = MagicMock()
        mock_bridge.enabled = True
        mock_service = MagicMock()
        mock_service.execute.return_value = HAMutationResult(
            HAOutcome.VERIFIED,
            "Verified light state.",
            "light.living_room",
            "light",
            "turn_on",
            "req-1",
            HARisk.SAFE,
        )

        with (
            patch("rex.openclaw.tools.ha_tool._get_ha_bridge", return_value=mock_bridge),
            patch("rex.openclaw.tools.ha_tool._get_mutation_service", return_value=mock_service),
        ):
            result = ha_call_service(
                "light",
                "turn_on",
                "light.living_room",
                context={"user_id": "james", "request_id": "req-1"},
            )

        mock_service.execute.assert_called_once()
        intent_arg = mock_service.execute.call_args.args[0]
        assert intent_arg.domain == "light"
        assert intent_arg.service == "turn_on"
        assert intent_arg.entity_id == "light.living_room"
        assert result["success"] is True
        assert result["status"] == "verified"

    def test_ha_call_service_with_data(self):
        """Extra data dict is merged into intent_data alongside entity_id."""
        from rex.openclaw.tools.ha_tool import ha_call_service

        mock_bridge = MagicMock()
        mock_bridge.enabled = True
        mock_service = MagicMock()
        from rex.ha.mutation_service import HAMutationResult, HAOutcome, HARisk

        mock_service.execute.return_value = HAMutationResult(
            HAOutcome.VERIFIED, "Done.", "light.bedroom", "light", "turn_on", "req-2", HARisk.SAFE
        )

        with (
            patch("rex.openclaw.tools.ha_tool._get_ha_bridge", return_value=mock_bridge),
            patch("rex.openclaw.tools.ha_tool._get_mutation_service", return_value=mock_service),
        ):
            ha_call_service(
                "light",
                "turn_on",
                "light.bedroom",
                data={"brightness_pct": 50},
                context={"user_id": "james", "request_id": "req-2"},
            )

        intent_arg = mock_service.execute.call_args.args[0]
        assert intent_arg.entity_id == "light.bedroom"
        assert intent_arg.parameters["brightness_pct"] == 50

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

    def test_ha_call_service_requires_identity_context(self):
        """Missing identity fails closed before mutation dispatch."""
        from rex.openclaw.tools.ha_tool import ha_call_service

        mock_bridge = MagicMock()
        mock_bridge.enabled = True

        with patch("rex.openclaw.tools.ha_tool._get_ha_bridge", return_value=mock_bridge):
            result = ha_call_service(
                "light",
                "turn_off",
                "light.office",
            )

        assert result["success"] is False
        assert result["status"] == "denied"
