"""Tests for rex.openclaw.policy_adapter — US-P3-009 and US-P3-010.

US-P3-009: block path — denied and requires-approval tools raise.
US-P3-010: allow path  — low-risk auto-execute tools pass without raising.
"""

from __future__ import annotations

import pytest

from rex.openclaw.policy_adapter import PolicyAdapter
from rex.openclaw.tool_executor import ApprovalRequiredError, PolicyDeniedError
from rex.policy import ActionPolicy
from rex.policy_engine import PolicyEngine


def _adapter(*extra_policies: ActionPolicy) -> PolicyAdapter:
    """Return a PolicyAdapter backed by a fresh engine with *extra_policies*."""
    engine = PolicyEngine(policies=list(extra_policies))
    return PolicyAdapter(engine=engine)


# ---------------------------------------------------------------------------
# US-P3-009: block path
# ---------------------------------------------------------------------------


class TestPolicyAdapterBlockPath:
    def test_guard_raises_approval_required_for_medium_risk(self):
        """guard() raises ApprovalRequiredError for a medium-risk tool."""
        adapter = _adapter()
        with pytest.raises(ApprovalRequiredError):
            adapter.guard("send_email")

    def test_guard_raises_approval_required_for_high_risk(self):
        """guard() raises ApprovalRequiredError for a high-risk tool."""
        adapter = _adapter()
        with pytest.raises(ApprovalRequiredError):
            adapter.guard("execute_command")

    def test_guard_raises_policy_denied_for_blocked_recipient(self):
        """guard() raises PolicyDeniedError when recipient is on the deny list."""
        from rex.contracts import RiskLevel

        policy = ActionPolicy(
            tool_name="send_email",
            risk=RiskLevel.MEDIUM,
            allow_auto=False,
            denied_recipients=["blocked@example.com"],
        )
        adapter = _adapter(policy)
        with pytest.raises(PolicyDeniedError):
            adapter.guard("send_email", metadata={"recipient": "blocked@example.com"})

    def test_guard_raises_policy_denied_for_blocked_domain(self):
        """guard() raises PolicyDeniedError when domain is on the deny list."""
        from rex.contracts import RiskLevel

        policy = ActionPolicy(
            tool_name="send_email",
            risk=RiskLevel.MEDIUM,
            allow_auto=False,
            denied_domains=["malicious.com"],
        )
        adapter = _adapter(policy)
        with pytest.raises(PolicyDeniedError):
            adapter.guard("send_email", metadata={"recipient": "user@malicious.com"})

    def test_check_returns_requires_approval_for_medium_risk(self):
        """check() returns a decision with requires_approval=True for medium-risk tools."""
        adapter = _adapter()
        decision = adapter.check("send_email")
        assert decision.requires_approval is True
        assert decision.denied is False
        assert decision.allowed is True

    def test_check_returns_denied_for_blocked_recipient(self):
        """check() returns denied=True when recipient is on the deny list."""
        from rex.contracts import RiskLevel

        policy = ActionPolicy(
            tool_name="send_email",
            risk=RiskLevel.MEDIUM,
            allow_auto=False,
            denied_recipients=["bad@example.com"],
        )
        adapter = _adapter(policy)
        decision = adapter.check("send_email", metadata={"recipient": "bad@example.com"})
        assert decision.denied is True
        assert decision.allowed is False

    def test_guard_custom_high_risk_tool(self):
        """guard() raises for a custom-registered high-risk tool."""
        from rex.contracts import RiskLevel

        policy = ActionPolicy(
            tool_name="my_dangerous_tool",
            risk=RiskLevel.HIGH,
            allow_auto=False,
        )
        adapter = _adapter(policy)
        with pytest.raises(ApprovalRequiredError):
            adapter.guard("my_dangerous_tool")

    def test_guard_unknown_tool_requires_approval_by_default(self):
        """Unknown tools use the medium-risk default policy — requires approval."""
        adapter = _adapter()
        with pytest.raises(ApprovalRequiredError):
            adapter.guard("totally_unknown_tool_xyz")


# ---------------------------------------------------------------------------
# US-P3-010: allow path
# ---------------------------------------------------------------------------


class TestPolicyAdapterAllowPath:
    def test_guard_does_not_raise_for_low_risk_auto_tool(self):
        """guard() does not raise for a low-risk, auto-execute tool."""
        adapter = _adapter()
        adapter.guard("time_now")  # should not raise

    def test_guard_does_not_raise_for_weather_now(self):
        """guard() does not raise for weather_now (low-risk, auto-execute)."""
        adapter = _adapter()
        adapter.guard("weather_now")

    def test_guard_does_not_raise_for_web_search(self):
        """guard() does not raise for web_search (low-risk, auto-execute)."""
        adapter = _adapter()
        adapter.guard("web_search")

    def test_check_returns_allowed_no_approval_for_low_risk(self):
        """check() returns allowed=True, requires_approval=False, denied=False for low-risk."""
        adapter = _adapter()
        decision = adapter.check("time_now")
        assert decision.allowed is True
        assert decision.requires_approval is False
        assert decision.denied is False

    def test_guard_custom_low_risk_auto_tool(self):
        """guard() does not raise for a custom low-risk, auto-execute tool."""
        from rex.contracts import RiskLevel

        policy = ActionPolicy(
            tool_name="my_safe_tool",
            risk=RiskLevel.LOW,
            allow_auto=True,
        )
        adapter = _adapter(policy)
        adapter.guard("my_safe_tool")  # should not raise

    def test_check_custom_low_risk_returns_no_approval(self):
        """check() returns requires_approval=False for a custom low-risk tool."""
        from rex.contracts import RiskLevel

        policy = ActionPolicy(
            tool_name="my_safe_tool",
            risk=RiskLevel.LOW,
            allow_auto=True,
        )
        adapter = _adapter(policy)
        decision = adapter.check("my_safe_tool")
        assert decision.requires_approval is False
        assert decision.denied is False

    def test_register_returns_none_without_openclaw(self):
        """register() returns None when openclaw is not installed."""
        adapter = _adapter()
        result = adapter.register()
        assert result is None

    def test_register_accepts_agent_arg(self):
        """register() accepts an agent argument without error."""
        adapter = _adapter()
        result = adapter.register(agent=object())
        assert result is None

    def test_engine_property_returns_policy_engine(self):
        """engine property returns the wrapped PolicyEngine instance."""
        from rex.policy_engine import PolicyEngine

        adapter = _adapter()
        assert isinstance(adapter.engine, PolicyEngine)
