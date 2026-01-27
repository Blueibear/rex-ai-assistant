"""Tests for the policy models and policy engine."""

from __future__ import annotations

import pytest

from rex.contracts import RiskLevel, ToolCall
from rex.policy import ActionPolicy, PolicyDecision
from rex.policy_engine import (
    DEFAULT_POLICIES,
    PolicyEngine,
    get_policy_engine,
    reset_policy_engine,
)


class TestActionPolicy:
    """Tests for ActionPolicy model."""

    def test_minimal_policy_creation(self):
        """Test creating a policy with minimal fields."""
        policy = ActionPolicy(tool_name="test_tool")
        assert policy.tool_name == "test_tool"
        assert policy.risk == RiskLevel.LOW
        assert policy.allow_auto is False
        assert policy.allowed_recipients is None
        assert policy.denied_recipients is None
        assert policy.allowed_domains is None
        assert policy.denied_domains is None

    def test_full_policy_creation(self):
        """Test creating a policy with all fields."""
        policy = ActionPolicy(
            tool_name="send_email",
            risk=RiskLevel.MEDIUM,
            allow_auto=False,
            allowed_recipients=["trusted@example.com"],
            denied_recipients=["spam@example.com"],
            allowed_domains=["example.com"],
            denied_domains=["malicious.com"],
        )
        assert policy.tool_name == "send_email"
        assert policy.risk == RiskLevel.MEDIUM
        assert policy.allow_auto is False
        assert policy.allowed_recipients == ["trusted@example.com"]
        assert policy.denied_recipients == ["spam@example.com"]
        assert policy.allowed_domains == ["example.com"]
        assert policy.denied_domains == ["malicious.com"]

    def test_policy_json_roundtrip(self):
        """Test that policies can be serialized and deserialized."""
        original = ActionPolicy(
            tool_name="test",
            risk=RiskLevel.HIGH,
            allow_auto=True,
            denied_domains=["blocked.com"],
        )
        json_str = original.model_dump_json()
        restored = ActionPolicy.model_validate_json(json_str)
        assert restored.tool_name == original.tool_name
        assert restored.risk == original.risk
        assert restored.allow_auto == original.allow_auto
        assert restored.denied_domains == original.denied_domains


class TestPolicyDecision:
    """Tests for PolicyDecision model."""

    def test_allowed_decision(self):
        """Test creating an allowed decision."""
        decision = PolicyDecision(
            allowed=True,
            reason="Test reason",
            requires_approval=False,
            denied=False,
        )
        assert decision.allowed is True
        assert decision.reason == "Test reason"
        assert decision.requires_approval is False
        assert decision.denied is False

    def test_approval_required_decision(self):
        """Test creating a decision requiring approval."""
        decision = PolicyDecision(
            allowed=True,
            reason="Medium-risk action",
            requires_approval=True,
            denied=False,
        )
        assert decision.allowed is True
        assert decision.requires_approval is True
        assert decision.denied is False

    def test_denied_decision(self):
        """Test creating a denied decision."""
        decision = PolicyDecision(
            allowed=False,
            reason="Recipient is blocked",
            requires_approval=False,
            denied=True,
        )
        assert decision.allowed is False
        assert decision.denied is True

    def test_decision_json_roundtrip(self):
        """Test that decisions can be serialized and deserialized."""
        original = PolicyDecision(
            allowed=True,
            reason="Test",
            requires_approval=True,
            denied=False,
        )
        json_str = original.model_dump_json()
        restored = PolicyDecision.model_validate_json(json_str)
        assert restored.allowed == original.allowed
        assert restored.reason == original.reason
        assert restored.requires_approval == original.requires_approval
        assert restored.denied == original.denied


class TestPolicyEngine:
    """Tests for PolicyEngine."""

    def test_default_policies_loaded(self):
        """Test that default policies are loaded on initialization."""
        engine = PolicyEngine()
        assert len(engine.policies) > 0
        assert "time_now" in engine.policies
        assert "send_email" in engine.policies

    def test_custom_policies_extend_defaults(self):
        """Test that custom policies are added to defaults."""
        custom_policy = ActionPolicy(
            tool_name="custom_tool",
            risk=RiskLevel.HIGH,
        )
        engine = PolicyEngine(policies=[custom_policy])
        assert "custom_tool" in engine.policies
        assert "time_now" in engine.policies  # Default still present

    def test_custom_policies_override_defaults(self):
        """Test that custom policies can override defaults."""
        custom_policy = ActionPolicy(
            tool_name="time_now",  # Override default
            risk=RiskLevel.HIGH,
            allow_auto=False,
        )
        engine = PolicyEngine(policies=[custom_policy])
        policy = engine.get_policy("time_now")
        assert policy.risk == RiskLevel.HIGH
        assert policy.allow_auto is False

    def test_get_policy_returns_default_for_unknown_tool(self):
        """Test that unknown tools get the default policy."""
        engine = PolicyEngine()
        policy = engine.get_policy("unknown_tool")
        assert policy.tool_name == "__default__"
        assert policy.risk == RiskLevel.MEDIUM
        assert policy.allow_auto is False


class TestPolicyEngineDecisions:
    """Tests for PolicyEngine.decide() method."""

    def test_low_risk_auto_execute_allowed(self):
        """Test that low-risk tools with allow_auto=True auto-execute."""
        engine = PolicyEngine()
        tool_call = ToolCall(tool="time_now", args={"location": "Dallas, TX"})
        decision = engine.decide(tool_call, {})
        assert decision.allowed is True
        assert decision.requires_approval is False
        assert decision.denied is False

    def test_medium_risk_requires_approval(self):
        """Test that medium-risk tools require approval."""
        engine = PolicyEngine()
        tool_call = ToolCall(tool="send_email", args={"to": "user@example.com"})
        decision = engine.decide(tool_call, {"recipient": "user@example.com"})
        assert decision.allowed is True
        assert decision.requires_approval is True
        assert decision.denied is False

    def test_high_risk_requires_approval(self):
        """Test that high-risk tools require approval."""
        engine = PolicyEngine()
        tool_call = ToolCall(tool="execute_command", args={"command": "ls"})
        decision = engine.decide(tool_call, {})
        assert decision.allowed is True
        assert decision.requires_approval is True
        assert decision.denied is False

    def test_denied_recipient_blocks_action(self):
        """Test that denied recipients block the action."""
        policy = ActionPolicy(
            tool_name="test_tool",
            risk=RiskLevel.LOW,
            allow_auto=True,
            denied_recipients=["blocked@example.com", "spam@test.com"],
        )
        engine = PolicyEngine(policies=[policy])
        tool_call = ToolCall(tool="test_tool", args={})
        decision = engine.decide(tool_call, {"recipient": "blocked@example.com"})
        assert decision.denied is True
        assert decision.allowed is False
        assert "deny list" in decision.reason.lower()

    def test_denied_domain_blocks_action(self):
        """Test that denied domains block the action."""
        policy = ActionPolicy(
            tool_name="test_tool",
            risk=RiskLevel.LOW,
            allow_auto=True,
            denied_domains=["malicious.com"],
        )
        engine = PolicyEngine(policies=[policy])
        tool_call = ToolCall(tool="test_tool", args={})
        decision = engine.decide(tool_call, {"domain": "malicious.com"})
        assert decision.denied is True
        assert "deny list" in decision.reason.lower()

    def test_domain_extracted_from_recipient(self):
        """Test that domain is extracted from recipient email."""
        policy = ActionPolicy(
            tool_name="test_tool",
            risk=RiskLevel.LOW,
            allow_auto=True,
            denied_domains=["blocked.com"],
        )
        engine = PolicyEngine(policies=[policy])
        tool_call = ToolCall(tool="test_tool", args={})
        decision = engine.decide(tool_call, {"recipient": "user@blocked.com"})
        assert decision.denied is True

    def test_allowed_recipients_whitelist(self):
        """Test that allowed_recipients acts as a whitelist."""
        policy = ActionPolicy(
            tool_name="test_tool",
            risk=RiskLevel.LOW,
            allow_auto=True,
            allowed_recipients=["trusted@example.com"],
        )
        engine = PolicyEngine(policies=[policy])
        tool_call = ToolCall(tool="test_tool", args={})

        # Allowed recipient passes
        decision = engine.decide(tool_call, {"recipient": "trusted@example.com"})
        assert decision.denied is False
        assert decision.allowed is True

        # Unlisted recipient is blocked
        decision = engine.decide(tool_call, {"recipient": "other@example.com"})
        assert decision.denied is True
        assert "not in the allowed list" in decision.reason.lower()

    def test_allowed_domains_whitelist(self):
        """Test that allowed_domains acts as a whitelist."""
        policy = ActionPolicy(
            tool_name="test_tool",
            risk=RiskLevel.LOW,
            allow_auto=True,
            allowed_domains=["company.com"],
        )
        engine = PolicyEngine(policies=[policy])
        tool_call = ToolCall(tool="test_tool", args={})

        # Allowed domain passes
        decision = engine.decide(tool_call, {"domain": "company.com"})
        assert decision.denied is False

        # Unlisted domain is blocked
        decision = engine.decide(tool_call, {"domain": "external.com"})
        assert decision.denied is True

    def test_case_insensitive_matching(self):
        """Test that recipient/domain matching is case-insensitive."""
        policy = ActionPolicy(
            tool_name="test_tool",
            risk=RiskLevel.LOW,
            allow_auto=True,
            denied_recipients=["Blocked@Example.COM"],
        )
        engine = PolicyEngine(policies=[policy])
        tool_call = ToolCall(tool="test_tool", args={})
        decision = engine.decide(tool_call, {"recipient": "blocked@example.com"})
        assert decision.denied is True

    def test_unknown_tool_uses_default_policy(self):
        """Test that unknown tools use the default policy."""
        engine = PolicyEngine()
        tool_call = ToolCall(tool="completely_unknown_tool", args={})
        decision = engine.decide(tool_call, {})
        # Default policy is medium risk, requires approval
        assert decision.allowed is True
        assert decision.requires_approval is True

    def test_empty_metadata_still_works(self):
        """Test that empty metadata doesn't cause errors."""
        engine = PolicyEngine()
        tool_call = ToolCall(tool="time_now", args={})
        decision = engine.decide(tool_call, {})
        assert decision.allowed is True

    def test_none_metadata_still_works(self):
        """Test that None metadata doesn't cause errors."""
        engine = PolicyEngine()
        tool_call = ToolCall(tool="time_now", args={})
        decision = engine.decide(tool_call, None)
        assert decision.allowed is True


class TestPolicyEngineManagement:
    """Tests for PolicyEngine policy management methods."""

    def test_add_policy(self):
        """Test adding a new policy."""
        engine = PolicyEngine()
        new_policy = ActionPolicy(
            tool_name="new_tool",
            risk=RiskLevel.HIGH,
        )
        engine.add_policy(new_policy)
        assert "new_tool" in engine.policies
        assert engine.get_policy("new_tool").risk == RiskLevel.HIGH

    def test_remove_policy(self):
        """Test removing a policy."""
        engine = PolicyEngine()
        assert "time_now" in engine.policies
        result = engine.remove_policy("time_now")
        assert result is True
        assert "time_now" not in engine.policies

    def test_remove_nonexistent_policy(self):
        """Test removing a policy that doesn't exist."""
        engine = PolicyEngine()
        result = engine.remove_policy("nonexistent")
        assert result is False


class TestPolicyEngineSingleton:
    """Tests for the singleton pattern."""

    def test_get_policy_engine_returns_singleton(self):
        """Test that get_policy_engine returns the same instance."""
        reset_policy_engine()
        engine1 = get_policy_engine()
        engine2 = get_policy_engine()
        assert engine1 is engine2

    def test_reset_policy_engine_creates_new_instance(self):
        """Test that reset_policy_engine allows a new instance."""
        engine1 = get_policy_engine()
        reset_policy_engine()
        engine2 = get_policy_engine()
        assert engine1 is not engine2


class TestDefaultPolicies:
    """Tests for the default policy definitions."""

    def test_default_policies_defined(self):
        """Test that default policies are defined."""
        assert len(DEFAULT_POLICIES) > 0

    def test_time_now_is_low_risk_auto(self):
        """Test that time_now is low risk with auto-execute."""
        policy = next(p for p in DEFAULT_POLICIES if p.tool_name == "time_now")
        assert policy.risk == RiskLevel.LOW
        assert policy.allow_auto is True

    def test_send_email_is_medium_risk(self):
        """Test that send_email is medium risk."""
        policy = next(p for p in DEFAULT_POLICIES if p.tool_name == "send_email")
        assert policy.risk == RiskLevel.MEDIUM
        assert policy.allow_auto is False

    def test_execute_command_is_high_risk(self):
        """Test that execute_command is high risk."""
        policy = next(p for p in DEFAULT_POLICIES if p.tool_name == "execute_command")
        assert policy.risk == RiskLevel.HIGH
        assert policy.allow_auto is False
