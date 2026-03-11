"""Tests for US-060: Tool execution error handling.

Verifies that tool failures are captured, recorded, and do not crash the assistant.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rex.audit import AuditLogger, reset_audit_logger
from rex.tool_router import (
    ApprovalRequiredError,
    CredentialMissingError,
    PolicyDeniedError,
    execute_tool,
)


@pytest.fixture(autouse=True)
def isolated_audit_logger(tmp_path):
    """Redirect the global audit logger to a temp path for test isolation."""
    from rex import audit as audit_module

    log_path = tmp_path / "test_audit.log"
    test_logger = AuditLogger(log_path=log_path)
    audit_module._default_audit_logger = test_logger
    yield test_logger
    reset_audit_logger()


class TestToolFailuresCaptured:
    """Tool failures are caught and returned as error results, not exceptions."""

    def test_exception_in_tool_returns_error_result(self, isolated_audit_logger):
        """Exceptions inside the tool body are caught and returned as errors."""
        with patch("rex.tool_router._execute_time_now", side_effect=RuntimeError("boom")):
            result = execute_tool(
                {"tool": "time_now", "args": {"location": "Dallas, TX"}},
                {},
                skip_policy_check=True,
            )

        assert "error" in result

    def test_unknown_tool_returns_error_not_exception(self, isolated_audit_logger):
        result = execute_tool(
            {"tool": "does_not_exist", "args": {}},
            {},
            skip_policy_check=True,
        )
        assert "error" in result

    def test_invalid_request_payload_returns_error(self, isolated_audit_logger):
        result = execute_tool("not a dict", {}, skip_policy_check=True)  # type: ignore[arg-type]
        assert "error" in result

    def test_missing_tool_name_returns_error(self, isolated_audit_logger):
        result = execute_tool({"args": {}}, {}, skip_policy_check=True)
        assert "error" in result

    def test_invalid_args_type_returns_error(self, isolated_audit_logger):
        result = execute_tool(
            {"tool": "time_now", "args": "not a dict"},
            {},
            skip_policy_check=True,
        )
        assert "error" in result

    def test_policy_denied_raises_not_swallowed(self, isolated_audit_logger):
        """PolicyDeniedError propagates to caller; it is NOT swallowed."""
        from unittest.mock import MagicMock

        denied = MagicMock()
        denied.denied = True
        denied.requires_approval = False
        denied.reason = "test denial"

        engine = MagicMock()
        engine.decide.return_value = denied

        with pytest.raises(PolicyDeniedError):
            execute_tool(
                {"tool": "time_now", "args": {"location": "Dallas, TX"}},
                {},
                policy_engine=engine,
            )

    def test_approval_required_raises_not_swallowed(self, isolated_audit_logger):
        from unittest.mock import MagicMock

        requires = MagicMock()
        requires.denied = False
        requires.requires_approval = True
        requires.reason = "needs approval"

        engine = MagicMock()
        engine.decide.return_value = requires

        with pytest.raises(ApprovalRequiredError):
            execute_tool(
                {"tool": "time_now", "args": {"location": "Dallas, TX"}},
                {},
                policy_engine=engine,
            )


class TestFailureReasonRecorded:
    """Failure reasons are stored in the audit log."""

    def test_exception_message_in_audit_log(self, isolated_audit_logger):
        with patch("rex.tool_router._execute_time_now", side_effect=ValueError("bad value")):
            execute_tool(
                {"tool": "time_now", "args": {"location": "Dallas, TX"}},
                {},
                skip_policy_check=True,
            )

        entries = isolated_audit_logger.read()
        assert len(entries) == 1
        assert entries[0].error is not None
        assert "bad value" in entries[0].error

    def test_unknown_tool_error_in_audit_log(self, isolated_audit_logger):
        execute_tool({"tool": "missing_tool", "args": {}}, {}, skip_policy_check=True)

        entries = isolated_audit_logger.read()
        assert len(entries) == 1
        assert entries[0].error is not None
        assert "missing_tool" in entries[0].error

    def test_policy_denied_error_in_audit_log(self, isolated_audit_logger):
        from unittest.mock import MagicMock

        denied = MagicMock()
        denied.denied = True
        denied.requires_approval = False
        denied.reason = "unauthorized operation"

        engine = MagicMock()
        engine.decide.return_value = denied

        with pytest.raises(PolicyDeniedError):
            execute_tool(
                {"tool": "time_now", "args": {"location": "Dallas, TX"}},
                {},
                policy_engine=engine,
            )

        entries = isolated_audit_logger.read()
        assert len(entries) == 1
        assert entries[0].policy_decision == "denied"
        assert entries[0].error is not None

    def test_error_result_contains_message(self, isolated_audit_logger):
        result = execute_tool(
            {"tool": "unknown_xyz", "args": {}},
            {},
            skip_policy_check=True,
        )
        # Error result must contain a human-readable message
        assert "error" in result
        error_payload = result["error"]
        assert isinstance(error_payload, dict)
        assert "message" in error_payload


class TestExecutionDoesNotCrashAssistant:
    """The assistant remains stable after tool failures."""

    def test_exception_in_tool_does_not_propagate(self, isolated_audit_logger):
        """RuntimeError inside a tool is caught; execute_tool returns a result."""
        with patch("rex.tool_router._execute_time_now", side_effect=RuntimeError("crash")):
            result = execute_tool(
                {"tool": "time_now", "args": {"location": "Dallas, TX"}},
                {},
                skip_policy_check=True,
            )
        # Must return a dict, not raise
        assert isinstance(result, dict)

    def test_multiple_failures_in_sequence_all_handled(self, isolated_audit_logger):
        """Successive failures each return error results without crashing."""
        for _ in range(3):
            result = execute_tool(
                {"tool": "nonexistent", "args": {}},
                {},
                skip_policy_check=True,
            )
            assert "error" in result

    def test_audit_log_failure_does_not_crash_execution(self, isolated_audit_logger):
        """Even if the audit logger raises on write, tool execution continues."""
        # Make the logger's log() method raise an exception
        isolated_audit_logger.log = lambda entry: (_ for _ in ()).throw(  # type: ignore[method-assign]
            OSError("disk full")
        )

        # Tool execution must succeed even with broken audit logger
        result = execute_tool(
            {"tool": "time_now", "args": {"location": "Dallas, TX"}},
            {"timezone": "America/Chicago"},
            skip_policy_check=True,
        )
        assert isinstance(result, dict)

    def test_route_if_tool_request_does_not_crash_on_tool_failure(self, isolated_audit_logger):
        """route_if_tool_request returns a safe string on exception in model_fn."""
        from rex.tool_router import route_if_tool_request

        def bad_model_fn(msg):
            raise RuntimeError("model blew up")

        line = 'TOOL_REQUEST: {"tool": "time_now", "args": {"location": "Dallas, TX"}}'
        result = route_if_tool_request(
            line,
            {"timezone": "America/Chicago"},
            bad_model_fn,
            skip_policy_check=True,
        )
        assert isinstance(result, str)
        assert len(result) > 0
