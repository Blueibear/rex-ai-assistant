"""Tests for US-P4-009: Verify HA commands execute correctly through OpenClaw.

Covers:
- PolicyAdapter.guard() raises ApprovalRequiredError for home_assistant_call_service
- PolicyAdapter.check() returns correct decision (requires_approval=True)
- RexAgent.call_tool() is blocked by policy before any HA service call
- ha_call_service() executes correctly when bridge is enabled (service-level)
- ha_call_service() returns error dict when bridge is not configured

home_assistant_call_service is explicitly in DEFAULT_POLICIES with
risk=RiskLevel.MEDIUM, allow_auto=False — so it always requires approval.
"""

from __future__ import annotations

import pytest

from rex.openclaw.policy_adapter import PolicyAdapter
from rex.policy import ActionPolicy
from rex.policy_engine import PolicyEngine
from rex.tool_router import ApprovalRequiredError, PolicyDeniedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adapter(*extra_policies: ActionPolicy) -> PolicyAdapter:
    """Return a PolicyAdapter backed by a fresh PolicyEngine."""
    engine = PolicyEngine(policies=list(extra_policies))
    return PolicyAdapter(engine=engine)


# ---------------------------------------------------------------------------
# PolicyAdapter.guard() — home_assistant_call_service is MEDIUM risk
# ---------------------------------------------------------------------------


class TestHaToolPolicyGuard:
    """PolicyAdapter.guard() raises for home_assistant_call_service."""

    def test_guard_ha_call_service_raises_approval_required(self):
        """home_assistant_call_service (explicit MEDIUM policy) raises ApprovalRequiredError."""
        adapter = _adapter()
        with pytest.raises(ApprovalRequiredError):
            adapter.guard("home_assistant_call_service")

    def test_guard_does_not_raise_with_low_risk_override(self):
        """With a LOW-risk allow_auto override, guard() does not raise."""
        from rex.contracts import RiskLevel

        low_risk = ActionPolicy(
            tool_name="home_assistant_call_service",
            risk=RiskLevel.LOW,
            allow_auto=True,
        )
        adapter = _adapter(low_risk)
        adapter.guard("home_assistant_call_service")  # must not raise


# ---------------------------------------------------------------------------
# PolicyAdapter.check() — decision fields
# ---------------------------------------------------------------------------


class TestHaToolPolicyCheck:
    """PolicyAdapter.check() returns the expected decision for HA tool."""

    def test_check_ha_call_service_requires_approval(self):
        """home_assistant_call_service: allowed=True, requires_approval=True, denied=False."""
        adapter = _adapter()
        decision = adapter.check("home_assistant_call_service")
        assert decision.allowed is True
        assert decision.requires_approval is True
        assert decision.denied is False


# ---------------------------------------------------------------------------
# RexAgent.call_tool() — policy gate raised before HA execution
# ---------------------------------------------------------------------------


class TestRexAgentCallToolHaGate:
    """RexAgent.call_tool() is blocked by policy adapter for HA tool."""

    def _make_agent(self, *extra_policies: ActionPolicy):
        from unittest.mock import MagicMock

        from rex.config import AppConfig
        from rex.openclaw.agent import RexAgent

        engine = PolicyEngine(policies=list(extra_policies))
        policy_adapter = PolicyAdapter(engine=engine)
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "ok"
        config = AppConfig(wakeword="rex")
        return RexAgent(llm=mock_llm, config=config, policy_adapter=policy_adapter)

    def test_call_tool_ha_raises_approval_required(self):
        """call_tool('home_assistant_call_service') raises before HA is called."""
        agent = self._make_agent()
        with pytest.raises(ApprovalRequiredError):
            agent.call_tool(
                "home_assistant_call_service",
                {"domain": "light", "service": "turn_on", "entity_id": "light.living_room"},
            )

    def test_call_tool_ha_service_not_called_when_policy_blocks(self):
        """The HA callable is never invoked when the policy adapter blocks."""
        from unittest.mock import MagicMock, patch

        from rex.config import AppConfig
        from rex.openclaw.agent import RexAgent

        engine = PolicyEngine(policies=[])
        policy_adapter = PolicyAdapter(engine=engine)
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "ok"
        config = AppConfig(wakeword="rex")
        agent = RexAgent(llm=mock_llm, config=config, policy_adapter=policy_adapter)

        with patch("rex.tool_router.execute_tool") as mock_execute:
            with pytest.raises(ApprovalRequiredError):
                agent.call_tool(
                    "home_assistant_call_service",
                    {"domain": "switch", "service": "turn_off", "entity_id": "switch.garage"},
                )
            mock_execute.assert_not_called()


# ---------------------------------------------------------------------------
# ha_call_service() — service-level execution (bridge mocked)
# ---------------------------------------------------------------------------


class TestHaCallServiceExecution:
    """ha_call_service executes correctly against a mocked HABridge."""

    def test_turn_on_light_succeeds(self):
        """ha_call_service returns success dict when bridge executes without error."""
        from unittest.mock import MagicMock, patch

        from rex.openclaw.tools.ha_tool import ha_call_service

        mock_bridge = MagicMock()
        mock_bridge.enabled = True
        mock_bridge._execute_intent.return_value = (True, "Light turn_on light.living_room.")

        with patch("rex.openclaw.tools.ha_tool._get_ha_bridge", return_value=mock_bridge):
            result = ha_call_service("light", "turn_on", "light.living_room")

        assert result["success"] is True
        assert result["entity_id"] == "light.living_room"
        assert "light.living_room" in result["message"]

    def test_turn_off_switch_succeeds(self):
        """ha_call_service returns success dict for a switch service call."""
        from unittest.mock import MagicMock, patch

        from rex.openclaw.tools.ha_tool import ha_call_service

        mock_bridge = MagicMock()
        mock_bridge.enabled = True
        mock_bridge._execute_intent.return_value = (True, "Switch turn_off switch.garage.")

        with patch("rex.openclaw.tools.ha_tool._get_ha_bridge", return_value=mock_bridge):
            result = ha_call_service("switch", "turn_off", "switch.garage")

        assert result["success"] is True
        assert result["entity_id"] == "switch.garage"

    def test_bridge_failure_propagated(self):
        """When _execute_intent returns (False, reason), result has success=False."""
        from unittest.mock import MagicMock, patch

        from rex.openclaw.tools.ha_tool import ha_call_service

        mock_bridge = MagicMock()
        mock_bridge.enabled = True
        mock_bridge._execute_intent.return_value = (False, "Entity not found.")

        with patch("rex.openclaw.tools.ha_tool._get_ha_bridge", return_value=mock_bridge):
            result = ha_call_service("light", "turn_on", "light.unknown")

        assert result["success"] is False
        assert "Entity not found" in result["message"]
        assert result["entity_id"] == "light.unknown"

    def test_not_configured_returns_error(self):
        """When bridge.enabled is False, ha_call_service returns an error dict."""
        from unittest.mock import MagicMock, patch

        from rex.openclaw.tools.ha_tool import ha_call_service

        mock_bridge = MagicMock()
        mock_bridge.enabled = False

        with patch("rex.openclaw.tools.ha_tool._get_ha_bridge", return_value=mock_bridge):
            result = ha_call_service("climate", "set_temperature", "climate.thermostat")

        assert result["success"] is False
        assert "not configured" in result["message"]
        mock_bridge._execute_intent.assert_not_called()

    def test_intent_fields_match_call_args(self):
        """IntentMatch constructed by ha_call_service has the correct fields."""
        from unittest.mock import MagicMock, patch

        from rex.openclaw.tools.ha_tool import ha_call_service

        mock_bridge = MagicMock()
        mock_bridge.enabled = True
        mock_bridge._execute_intent.return_value = (True, "Done.")

        with patch("rex.openclaw.tools.ha_tool._get_ha_bridge", return_value=mock_bridge):
            ha_call_service(
                "media_player",
                "play_media",
                "media_player.living_room",
                data={"media_content_id": "spotify:track:abc"},
            )

        intent_arg = mock_bridge._execute_intent.call_args.args[0]
        assert intent_arg.domain == "media_player"
        assert intent_arg.service == "play_media"
        assert intent_arg.entity_id == "media_player.living_room"
        assert intent_arg.data["media_content_id"] == "spotify:track:abc"
        assert intent_arg.source == "openclaw"
