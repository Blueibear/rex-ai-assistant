"""Tests for US-059: Tool execution logging.

Verifies that tool executions are logged with timestamps and parameters.
"""

from __future__ import annotations

import pytest

from rex.audit import AuditLogger, reset_audit_logger
from rex.tool_router import execute_tool


@pytest.fixture(autouse=True)
def isolated_audit_logger(tmp_path):
    """Redirect the global audit logger to a temp path for test isolation."""
    from rex import audit as audit_module

    log_path = tmp_path / "test_audit.log"
    test_logger = AuditLogger(log_path=log_path)
    audit_module._default_audit_logger = test_logger
    yield test_logger
    reset_audit_logger()


class TestToolExecutionLogged:
    """Tool executions are written to the audit log."""

    def test_successful_execution_logged(self, isolated_audit_logger):
        request = {"tool": "time_now", "args": {"location": "Dallas, TX"}}
        execute_tool(request, {"timezone": "America/Chicago"}, skip_policy_check=True)

        entries = isolated_audit_logger.read()
        assert len(entries) == 1
        assert entries[0].tool == "time_now"

    def test_unknown_tool_logged(self, isolated_audit_logger):
        request = {"tool": "nonexistent_tool", "args": {}}
        execute_tool(request, {}, skip_policy_check=True)

        entries = isolated_audit_logger.read()
        assert len(entries) == 1
        assert entries[0].tool == "nonexistent_tool"
        assert entries[0].error is not None

    def test_skip_audit_log_suppresses_logging(self, isolated_audit_logger):
        request = {"tool": "time_now", "args": {"location": "Dallas, TX"}}
        execute_tool(
            request, {"timezone": "America/Chicago"}, skip_policy_check=True, skip_audit_log=True
        )

        entries = isolated_audit_logger.read()
        assert len(entries) == 0

    def test_multiple_executions_all_logged(self, isolated_audit_logger):
        for _ in range(3):
            execute_tool(
                {"tool": "time_now", "args": {"location": "Dallas, TX"}},
                {"timezone": "America/Chicago"},
                skip_policy_check=True,
            )

        entries = isolated_audit_logger.read()
        assert len(entries) == 3


class TestExecutionTimestampsRecorded:
    """Each log entry contains a UTC timestamp."""

    def test_timestamp_present(self, isolated_audit_logger):
        from datetime import timezone

        request = {"tool": "time_now", "args": {"location": "Dallas, TX"}}
        execute_tool(request, {"timezone": "America/Chicago"}, skip_policy_check=True)

        entries = isolated_audit_logger.read()
        assert len(entries) == 1
        ts = entries[0].timestamp
        assert ts is not None
        assert ts.tzinfo is not None
        # Timestamp must be UTC-aware
        assert ts.tzinfo == timezone.utc or ts.utcoffset().total_seconds() == 0

    def test_timestamp_is_recent(self, isolated_audit_logger):
        from datetime import datetime, timedelta, timezone

        before = datetime.now(timezone.utc) - timedelta(seconds=5)
        request = {"tool": "time_now", "args": {"location": "Dallas, TX"}}
        execute_tool(request, {"timezone": "America/Chicago"}, skip_policy_check=True)
        after = datetime.now(timezone.utc) + timedelta(seconds=5)

        entries = isolated_audit_logger.read()
        assert len(entries) == 1
        ts = entries[0].timestamp
        assert before <= ts <= after

    def test_duration_ms_recorded(self, isolated_audit_logger):
        request = {"tool": "time_now", "args": {"location": "Dallas, TX"}}
        execute_tool(request, {"timezone": "America/Chicago"}, skip_policy_check=True)

        entries = isolated_audit_logger.read()
        assert len(entries) == 1
        assert entries[0].duration_ms is not None
        assert entries[0].duration_ms >= 0


class TestToolParametersStored:
    """Tool arguments are stored in the log entry."""

    def test_args_stored_in_log(self, isolated_audit_logger):
        request = {"tool": "time_now", "args": {"location": "Dallas, TX"}}
        execute_tool(request, {"timezone": "America/Chicago"}, skip_policy_check=True)

        entries = isolated_audit_logger.read()
        assert len(entries) == 1
        assert entries[0].tool_call_args.get("location") == "Dallas, TX"

    def test_empty_args_stored(self, isolated_audit_logger):
        request = {"tool": "nonexistent_tool", "args": {}}
        execute_tool(request, {}, skip_policy_check=True)

        entries = isolated_audit_logger.read()
        assert len(entries) == 1
        assert entries[0].tool_call_args == {}

    def test_multiple_args_stored(self, isolated_audit_logger):
        """All provided args are captured in the log."""
        request = {"tool": "web_search", "args": {"query": "test", "num_results": 5}}
        execute_tool(request, {}, skip_policy_check=True)

        entries = isolated_audit_logger.read()
        assert len(entries) == 1
        stored_args = entries[0].tool_call_args
        assert stored_args.get("query") == "test"
        assert stored_args.get("num_results") == 5

    def test_action_id_unique_per_execution(self, isolated_audit_logger):
        """Every execution gets a unique action ID."""
        for _ in range(3):
            execute_tool(
                {"tool": "time_now", "args": {"location": "Dallas, TX"}},
                {"timezone": "America/Chicago"},
                skip_policy_check=True,
            )

        entries = isolated_audit_logger.read()
        ids = [e.action_id for e in entries]
        assert len(set(ids)) == 3, "Each execution must have a unique action_id"

    def test_requested_by_stored_when_provided(self, isolated_audit_logger):
        request = {"tool": "time_now", "args": {"location": "Dallas, TX"}}
        execute_tool(
            request,
            {"timezone": "America/Chicago"},
            skip_policy_check=True,
            requested_by="test_user",
        )

        entries = isolated_audit_logger.read()
        assert len(entries) == 1
        assert entries[0].requested_by == "test_user"

    def test_policy_decision_recorded(self, isolated_audit_logger):
        request = {"tool": "time_now", "args": {"location": "Dallas, TX"}}
        execute_tool(request, {"timezone": "America/Chicago"}, skip_policy_check=True)

        entries = isolated_audit_logger.read()
        assert len(entries) == 1
        assert entries[0].policy_decision == "allowed"
