"""Tests for policy integration in the tool router."""

from __future__ import annotations

import pytest

from rex.contracts import RiskLevel
from rex.policy import ActionPolicy
from rex.policy_engine import PolicyEngine, reset_policy_engine
from rex.tool_router import (
    ApprovalRequiredError,
    PolicyDeniedError,
    execute_tool,
    route_if_tool_request,
)


class TestExecuteToolPolicyIntegration:
    """Tests for execute_tool policy checks."""

    def setup_method(self):
        """Reset the policy engine before each test."""
        reset_policy_engine()

    def test_low_risk_tool_auto_executes(self):
        """Test that low-risk auto-execute tools work without errors."""
        result = execute_tool(
            {"tool": "time_now", "args": {"location": "Dallas, TX"}},
            {},
        )
        assert "local_time" in result
        assert "timezone" in result

    def test_medium_risk_tool_raises_approval_required(self):
        """Test that medium-risk tools raise ApprovalRequiredError."""
        with pytest.raises(ApprovalRequiredError) as exc_info:
            execute_tool(
                {"tool": "send_email", "args": {"to": "user@example.com"}},
                {},
            )
        assert exc_info.value.tool == "send_email"
        assert "approval" in exc_info.value.reason.lower()

    def test_high_risk_tool_raises_approval_required(self):
        """Test that high-risk tools raise ApprovalRequiredError."""
        with pytest.raises(ApprovalRequiredError) as exc_info:
            execute_tool(
                {"tool": "execute_command", "args": {"command": "ls"}},
                {},
            )
        assert exc_info.value.tool == "execute_command"

    def test_denied_recipient_raises_policy_denied(self):
        """Test that denied recipients raise PolicyDeniedError."""
        policy = ActionPolicy(
            tool_name="test_tool",
            risk=RiskLevel.LOW,
            allow_auto=True,
            denied_recipients=["blocked@example.com"],
        )
        engine = PolicyEngine(policies=[policy])

        with pytest.raises(PolicyDeniedError) as exc_info:
            execute_tool(
                {"tool": "test_tool", "args": {"to": "blocked@example.com"}},
                {},
                policy_engine=engine,
            )
        assert exc_info.value.tool == "test_tool"
        assert "deny list" in exc_info.value.reason.lower()

    def test_denied_domain_raises_policy_denied(self):
        """Test that denied domains raise PolicyDeniedError."""
        policy = ActionPolicy(
            tool_name="web_request",
            risk=RiskLevel.LOW,
            allow_auto=True,
            denied_domains=["malicious.com"],
        )
        engine = PolicyEngine(policies=[policy])

        with pytest.raises(PolicyDeniedError) as exc_info:
            execute_tool(
                {"tool": "web_request", "args": {"url": "https://malicious.com/page"}},
                {},
                policy_engine=engine,
            )
        assert exc_info.value.tool == "web_request"

    def test_skip_policy_check_bypasses_approval(self):
        """Test that skip_policy_check=True bypasses policy checks."""
        # This would normally require approval, but we skip the check
        result = execute_tool(
            {"tool": "send_email", "args": {"to": "user@example.com"}},
            {},
            skip_policy_check=True,
        )
        # send_email is not implemented (unknown tool), so we get an error
        # but not a policy error - we bypassed the policy check
        assert "error" in result
        assert "Unknown tool" in result["error"]["message"]

    def test_custom_policy_engine_used(self):
        """Test that a custom policy engine is used when provided."""
        # Create a policy that auto-allows send_email
        policy = ActionPolicy(
            tool_name="send_email",
            risk=RiskLevel.LOW,
            allow_auto=True,
        )
        engine = PolicyEngine(policies=[policy])

        # This would normally require approval, but our custom policy auto-allows
        result = execute_tool(
            {"tool": "send_email", "args": {"to": "user@example.com"}},
            {},
            policy_engine=engine,
        )
        # send_email is not implemented (unknown tool), but we get past policy check
        assert "error" in result
        assert "Unknown tool" in result["error"]["message"]

    def test_unknown_tool_uses_default_policy(self):
        """Test that unknown tools use the default (medium-risk) policy."""
        with pytest.raises(ApprovalRequiredError) as exc_info:
            execute_tool(
                {"tool": "completely_unknown_tool", "args": {}},
                {},
            )
        assert exc_info.value.tool == "completely_unknown_tool"


class TestRouteIfToolRequestPolicyIntegration:
    """Tests for route_if_tool_request policy integration."""

    def setup_method(self):
        """Reset the policy engine before each test."""
        reset_policy_engine()

    def test_low_risk_tool_routes_successfully(self):
        """Test that low-risk auto-execute tools route successfully."""
        calls = []

        def model_call(message):
            calls.append(message)
            return "final response"

        llm_text = 'TOOL_REQUEST: {"tool":"time_now","args":{"location":"Dallas, TX"}}'
        result = route_if_tool_request(llm_text, {}, model_call)

        assert result == "final response"
        assert len(calls) == 1
        assert "TOOL_RESULT" in calls[0]["content"]

    def test_policy_denied_returns_message(self):
        """Test that denied tools return a user-friendly message."""
        policy = ActionPolicy(
            tool_name="test_tool",
            risk=RiskLevel.LOW,
            allow_auto=True,
            denied_recipients=["blocked@example.com"],
        )
        engine = PolicyEngine(policies=[policy])

        def model_call(message):
            return "should not be called"

        llm_text = 'TOOL_REQUEST: {"tool":"test_tool","args":{"to":"blocked@example.com"}}'
        result = route_if_tool_request(llm_text, {}, model_call, policy_engine=engine)

        assert "cannot execute" in result.lower()
        assert "deny list" in result.lower()

    def test_approval_required_returns_message(self):
        """Test that approval-required tools return a user-friendly message."""

        def model_call(message):
            return "should not be called"

        llm_text = 'TOOL_REQUEST: {"tool":"send_email","args":{"to":"user@example.com"}}'
        result = route_if_tool_request(llm_text, {}, model_call)

        assert "requires your approval" in result.lower()

    def test_skip_policy_check_routes_normally(self):
        """Test that skip_policy_check allows routing even for approval-required tools."""
        calls = []

        def model_call(message):
            calls.append(message)
            return "final response"

        llm_text = 'TOOL_REQUEST: {"tool":"send_email","args":{"to":"user@example.com"}}'
        result = route_if_tool_request(
            llm_text, {}, model_call, skip_policy_check=True
        )

        assert result == "final response"
        assert len(calls) == 1
        # send_email is not implemented (unknown tool), so result contains error
        assert "Unknown tool" in calls[0]["content"]

    def test_non_tool_request_returns_unchanged(self):
        """Test that non-tool requests are returned unchanged."""
        result = route_if_tool_request("Hello, how are you?", {}, lambda x: "unused")
        assert result == "Hello, how are you?"


class TestMetadataExtraction:
    """Tests for metadata extraction from tool arguments."""

    def setup_method(self):
        """Reset the policy engine before each test."""
        reset_policy_engine()

    def test_to_field_extracted_as_recipient(self):
        """Test that 'to' field is extracted as recipient."""
        policy = ActionPolicy(
            tool_name="test",
            risk=RiskLevel.LOW,
            allow_auto=True,
            denied_recipients=["blocked@test.com"],
        )
        engine = PolicyEngine(policies=[policy])

        with pytest.raises(PolicyDeniedError):
            execute_tool(
                {"tool": "test", "args": {"to": "blocked@test.com"}},
                {},
                policy_engine=engine,
            )

    def test_recipient_field_extracted(self):
        """Test that 'recipient' field is extracted."""
        policy = ActionPolicy(
            tool_name="test",
            risk=RiskLevel.LOW,
            allow_auto=True,
            denied_recipients=["blocked@test.com"],
        )
        engine = PolicyEngine(policies=[policy])

        with pytest.raises(PolicyDeniedError):
            execute_tool(
                {"tool": "test", "args": {"recipient": "blocked@test.com"}},
                {},
                policy_engine=engine,
            )

    def test_url_domain_extracted(self):
        """Test that domain is extracted from URL."""
        policy = ActionPolicy(
            tool_name="test",
            risk=RiskLevel.LOW,
            allow_auto=True,
            denied_domains=["blocked.com"],
        )
        engine = PolicyEngine(policies=[policy])

        with pytest.raises(PolicyDeniedError):
            execute_tool(
                {"tool": "test", "args": {"url": "https://blocked.com/path/page"}},
                {},
                policy_engine=engine,
            )

    def test_url_domain_with_port_extracted(self):
        """Test that domain is extracted from URL with port."""
        policy = ActionPolicy(
            tool_name="test",
            risk=RiskLevel.LOW,
            allow_auto=True,
            denied_domains=["blocked.com"],
        )
        engine = PolicyEngine(policies=[policy])

        with pytest.raises(PolicyDeniedError):
            execute_tool(
                {"tool": "test", "args": {"url": "https://blocked.com:8080/path"}},
                {},
                policy_engine=engine,
            )

    def test_domain_field_extracted(self):
        """Test that 'domain' field is extracted directly."""
        policy = ActionPolicy(
            tool_name="test",
            risk=RiskLevel.LOW,
            allow_auto=True,
            denied_domains=["blocked.com"],
        )
        engine = PolicyEngine(policies=[policy])

        with pytest.raises(PolicyDeniedError):
            execute_tool(
                {"tool": "test", "args": {"domain": "blocked.com"}},
                {},
                policy_engine=engine,
            )

    def test_allowed_recipients_requires_recipient(self):
        """Test that allowed_recipients denies when no recipient provided."""
        policy = ActionPolicy(
            tool_name="test",
            risk=RiskLevel.LOW,
            allow_auto=True,
            allowed_recipients=["allowed@test.com"],
        )
        engine = PolicyEngine(policies=[policy])

        with pytest.raises(PolicyDeniedError):
            execute_tool(
                {"tool": "test", "args": {}},
                {},
                policy_engine=engine,
            )

    def test_allowed_domains_requires_domain(self):
        """Test that allowed_domains denies when no domain provided."""
        policy = ActionPolicy(
            tool_name="test",
            risk=RiskLevel.LOW,
            allow_auto=True,
            allowed_domains=["allowed.com"],
        )
        engine = PolicyEngine(policies=[policy])

        with pytest.raises(PolicyDeniedError):
            execute_tool(
                {"tool": "test", "args": {}},
                {},
                policy_engine=engine,
            )


class TestExceptionDetails:
    """Tests for exception details."""

    def test_policy_denied_error_attributes(self):
        """Test PolicyDeniedError has correct attributes."""
        error = PolicyDeniedError("test_tool", "Test reason")
        assert error.tool == "test_tool"
        assert error.reason == "Test reason"
        assert "test_tool" in str(error)
        assert "Test reason" in str(error)

    def test_approval_required_error_attributes(self):
        """Test ApprovalRequiredError has correct attributes."""
        error = ApprovalRequiredError("test_tool", "Needs approval")
        assert error.tool == "test_tool"
        assert error.reason == "Needs approval"
        assert "test_tool" in str(error)
        assert "Needs approval" in str(error)
