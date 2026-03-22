"""Tests for US-P4-007: Confirm policy gating works for policy-gated tools.

Verifies that send_email, send_sms, and calendar_create are correctly
gated by the PolicyAdapter and RexAgent.call_tool() before any service
call is made.

send_email     — explicit MEDIUM-risk policy in DEFAULT_POLICIES
send_sms       — unknown tool → falls back to default MEDIUM-risk policy
calendar_create — unknown tool (DEFAULT_POLICIES uses "calendar_create_event")
                  → falls back to default MEDIUM-risk policy

All three should raise ApprovalRequiredError when called without approval.
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
# PolicyAdapter.guard() — MEDIUM-risk approval gate
# ---------------------------------------------------------------------------


class TestPolicyGateGuard:
    """PolicyAdapter.guard() raises for every policy-gated tool."""

    def test_guard_send_email_raises_approval_required(self):
        """send_email (explicit MEDIUM policy) raises ApprovalRequiredError."""
        adapter = _adapter()
        with pytest.raises(ApprovalRequiredError):
            adapter.guard("send_email")

    def test_guard_send_sms_raises_approval_required(self):
        """send_sms (no explicit policy → MEDIUM default) raises ApprovalRequiredError."""
        adapter = _adapter()
        with pytest.raises(ApprovalRequiredError):
            adapter.guard("send_sms")

    def test_guard_calendar_create_raises_approval_required(self):
        """calendar_create (no explicit policy → MEDIUM default) raises ApprovalRequiredError."""
        adapter = _adapter()
        with pytest.raises(ApprovalRequiredError):
            adapter.guard("calendar_create")

    def test_guard_does_not_raise_when_low_risk_override_for_send_email(self):
        """With a LOW-risk, allow_auto override, guard() does not raise for send_email."""
        from rex.contracts import RiskLevel

        low_risk = ActionPolicy(
            tool_name="send_email",
            risk=RiskLevel.LOW,
            allow_auto=True,
        )
        adapter = _adapter(low_risk)
        adapter.guard("send_email")  # must not raise

    def test_guard_does_not_raise_when_low_risk_override_for_send_sms(self):
        """With a LOW-risk, allow_auto override, guard() does not raise for send_sms."""
        from rex.contracts import RiskLevel

        low_risk = ActionPolicy(
            tool_name="send_sms",
            risk=RiskLevel.LOW,
            allow_auto=True,
        )
        adapter = _adapter(low_risk)
        adapter.guard("send_sms")  # must not raise

    def test_guard_does_not_raise_when_low_risk_override_for_calendar_create(self):
        """With a LOW-risk, allow_auto override, guard() does not raise for calendar_create."""
        from rex.contracts import RiskLevel

        low_risk = ActionPolicy(
            tool_name="calendar_create",
            risk=RiskLevel.LOW,
            allow_auto=True,
        )
        adapter = _adapter(low_risk)
        adapter.guard("calendar_create")  # must not raise


# ---------------------------------------------------------------------------
# PolicyAdapter.check() — decision fields
# ---------------------------------------------------------------------------


class TestPolicyGateCheck:
    """PolicyAdapter.check() returns the expected decision for each tool."""

    def test_check_send_email_requires_approval(self):
        """send_email check: allowed=True, requires_approval=True, denied=False."""
        adapter = _adapter()
        decision = adapter.check("send_email")
        assert decision.allowed is True
        assert decision.requires_approval is True
        assert decision.denied is False

    def test_check_send_sms_requires_approval(self):
        """send_sms check: allowed=True, requires_approval=True, denied=False."""
        adapter = _adapter()
        decision = adapter.check("send_sms")
        assert decision.allowed is True
        assert decision.requires_approval is True
        assert decision.denied is False

    def test_check_calendar_create_requires_approval(self):
        """calendar_create check: allowed=True, requires_approval=True, denied=False."""
        adapter = _adapter()
        decision = adapter.check("calendar_create")
        assert decision.allowed is True
        assert decision.requires_approval is True
        assert decision.denied is False

    def test_check_send_email_denied_when_blocked_recipient(self):
        """send_email check: denied when recipient is on deny list."""
        from rex.contracts import RiskLevel

        policy = ActionPolicy(
            tool_name="send_email",
            risk=RiskLevel.MEDIUM,
            allow_auto=False,
            denied_recipients=["blocked@corp.com"],
        )
        adapter = _adapter(policy)
        decision = adapter.check("send_email", metadata={"recipient": "blocked@corp.com"})
        assert decision.denied is True
        assert decision.allowed is False

    def test_check_send_email_denied_when_blocked_domain(self):
        """send_email check: denied when domain is on deny list."""
        from rex.contracts import RiskLevel

        policy = ActionPolicy(
            tool_name="send_email",
            risk=RiskLevel.MEDIUM,
            allow_auto=False,
            denied_domains=["spam.com"],
        )
        adapter = _adapter(policy)
        decision = adapter.check("send_email", metadata={"recipient": "user@spam.com"})
        assert decision.denied is True
        assert decision.allowed is False


# ---------------------------------------------------------------------------
# RexAgent.call_tool() — policy gate raised before tool execution
# ---------------------------------------------------------------------------


class TestRexAgentCallToolPolicyGate:
    """RexAgent.call_tool() is blocked by the policy adapter for MEDIUM-risk tools."""

    def _make_agent(self, *extra_policies: ActionPolicy):
        """Return a RexAgent with injected minimal config and policy engine."""
        from unittest.mock import MagicMock

        from rex.config import AppConfig
        from rex.openclaw.agent import RexAgent

        engine = PolicyEngine(policies=list(extra_policies))
        policy_adapter = PolicyAdapter(engine=engine)

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "ok"

        config = AppConfig(wakeword="rex")
        return RexAgent(
            llm=mock_llm,
            config=config,
            policy_adapter=policy_adapter,
        )

    def test_call_tool_send_email_raises_approval_required(self):
        """call_tool('send_email') raises before the service is called."""
        agent = self._make_agent()
        with pytest.raises(ApprovalRequiredError):
            agent.call_tool("send_email", {"to": "user@example.com", "subject": "Hi", "body": "."})

    def test_call_tool_send_sms_raises_approval_required(self):
        """call_tool('send_sms') raises before the service is called."""
        agent = self._make_agent()
        with pytest.raises(ApprovalRequiredError):
            agent.call_tool("send_sms", {"to": "+15551234567", "body": "Hi"})

    def test_call_tool_calendar_create_raises_approval_required(self):
        """call_tool('calendar_create') raises before the service is called."""
        agent = self._make_agent()
        with pytest.raises(ApprovalRequiredError):
            agent.call_tool(
                "calendar_create",
                {"title": "Meeting", "start_time": "2026-04-01T09:00:00", "end_time": "2026-04-01T10:00:00"},
            )

    def test_call_tool_raises_policy_denied_for_blocked_recipient(self):
        """call_tool raises PolicyDeniedError when recipient is on the deny list."""
        from rex.contracts import RiskLevel

        policy = ActionPolicy(
            tool_name="send_email",
            risk=RiskLevel.MEDIUM,
            allow_auto=False,
            denied_recipients=["bad@example.com"],
        )
        agent = self._make_agent(policy)
        with pytest.raises(PolicyDeniedError):
            agent.call_tool(
                "send_email",
                {"to": "bad@example.com", "subject": "Test", "body": "body"},
                metadata={"recipient": "bad@example.com"},
            )

    def test_call_tool_service_not_called_when_policy_blocks(self):
        """The tool callable is never invoked when the policy adapter blocks."""
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
                agent.call_tool("send_email", {"to": "x@example.com", "subject": "S", "body": "B"})
            mock_execute.assert_not_called()
