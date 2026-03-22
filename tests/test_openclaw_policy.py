"""Tests for rex.openclaw.policy_adapter — US-P3-009 and US-P3-010.

US-P3-009: block path — denied and requires-approval tools raise.
US-P3-010: allow path  — low-risk auto-execute tools pass without raising.
"""

from __future__ import annotations

import pytest

from rex.openclaw.policy_adapter import PolicyAdapter
from rex.policy import ActionPolicy
from rex.policy_engine import PolicyEngine
from rex.tool_router import ApprovalRequiredError, PolicyDeniedError


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
