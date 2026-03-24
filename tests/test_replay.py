"""Tests for Rex replay functionality."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytest.importorskip("pydantic")

from rex.audit import LogEntry
from rex.audit import replay as audit_replay
from rex.contracts import ToolCall
from rex.replay import (
    ReplayResult,
    batch_replay,
    reconstruct_tool_call,
    replay,
)


class TestReconstructToolCall:
    """Tests for reconstructing ToolCall from LogEntry."""

    def test_basic_reconstruction(self):
        """Should reconstruct basic ToolCall from LogEntry."""
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={"location": "Dallas, TX"},
            policy_decision="allowed",
        )
        tool_call = reconstruct_tool_call(entry)

        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool == "time_now"
        assert tool_call.args == {"location": "Dallas, TX"}
        assert tool_call.requested_by == "replay:act_001"
        assert tool_call.idempotency_key == "replay-act_001"

    def test_reconstruction_with_empty_args(self):
        """Should handle empty tool args."""
        entry = LogEntry(
            action_id="act_002",
            tool="simple_tool",
            tool_call_args={},
            policy_decision="allowed",
        )
        tool_call = reconstruct_tool_call(entry)

        assert tool_call.tool == "simple_tool"
        assert tool_call.args == {}

    def test_reconstruction_with_complex_args(self):
        """Should preserve complex argument structures."""
        complex_args = {
            "nested": {
                "key": "value",
                "list": [1, 2, 3],
            },
            "flag": True,
            "count": 42,
        }
        entry = LogEntry(
            action_id="act_003",
            tool="complex_tool",
            tool_call_args=complex_args,
            policy_decision="allowed",
        )
        tool_call = reconstruct_tool_call(entry)

        assert tool_call.args == complex_args

    def test_reconstruction_timestamp_is_current(self):
        """Reconstructed ToolCall should have current timestamp."""
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        entry = LogEntry(
            timestamp=old_time,
            action_id="act_004",
            tool="test_tool",
            tool_call_args={},
            policy_decision="allowed",
        )
        tool_call = reconstruct_tool_call(entry)

        # The reconstructed tool call should have a recent timestamp
        assert tool_call.created_at > old_time


class TestReplay:
    """Tests for the replay function."""

    def test_replay_returns_result(self):
        """Replay should return a ReplayResult."""
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={"location": "Dallas"},
            policy_decision="allowed",
            tool_result={"local_time": "2024-01-15 10:30"},
        )
        result = replay(entry)

        assert isinstance(result, ReplayResult)

    def test_replay_preserves_original_entry(self):
        """ReplayResult should contain the original entry."""
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={"location": "Dallas"},
            policy_decision="allowed",
        )
        result = replay(entry)

        assert result.original_entry is entry

    def test_replay_contains_reconstructed_tool_call(self):
        """ReplayResult should contain reconstructed ToolCall."""
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={"location": "Dallas"},
            policy_decision="allowed",
        )
        result = replay(entry)

        assert isinstance(result.replayed_tool_call, ToolCall)
        assert result.replayed_tool_call.tool == "time_now"
        assert result.replayed_tool_call.args == {"location": "Dallas"}

    def test_replay_dry_run_default(self):
        """Replay should default to dry_run=True."""
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={},
            policy_decision="allowed",
        )
        result = replay(entry)

        assert result.dry_run is True

    def test_replay_dry_run_explicit_false(self):
        """Replay should accept dry_run=False."""
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={},
            policy_decision="allowed",
        )
        result = replay(entry, dry_run=False)

        assert result.dry_run is False

    def test_replay_has_stub_result(self):
        """Replay should return stub result indicating not implemented."""
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={"location": "Dallas"},
            policy_decision="allowed",
        )
        result = replay(entry)

        assert result.new_result is not None
        assert result.new_result["status"] == "stub"
        assert "not yet implemented" in result.new_result["message"].lower()

    def test_replay_has_comparison(self):
        """ReplayResult should contain comparison data."""
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={},
            policy_decision="allowed",
            tool_result={"local_time": "2024-01-15 10:30"},
        )
        result = replay(entry)

        assert "original_result" in result.comparison
        assert "new_result" in result.comparison
        assert result.comparison["original_result"] == {"local_time": "2024-01-15 10:30"}

    def test_replay_has_notes(self):
        """ReplayResult should contain explanatory notes."""
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={},
            policy_decision="allowed",
        )
        result = replay(entry)

        assert result.notes is not None
        assert "STUB" in result.notes
        assert "future phase" in result.notes.lower()

    def test_audit_replay_wrapper(self):
        """Audit replay wrapper should return a ReplayResult."""
        entry = LogEntry(
            action_id="act_006",
            tool="time_now",
            tool_call_args={"location": "Dallas"},
            policy_decision="allowed",
        )
        result = audit_replay(entry)
        assert isinstance(result, ReplayResult)

    def test_replay_has_timestamp(self):
        """ReplayResult should have replayed_at timestamp."""
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={},
            policy_decision="allowed",
        )
        result = replay(entry)

        assert isinstance(result.replayed_at, datetime)
        # Should be recent
        assert (datetime.now(timezone.utc) - result.replayed_at).total_seconds() < 5

    def test_replay_does_not_raise(self):
        """Replay should not raise exceptions for valid entries."""
        entry = LogEntry(
            action_id="act_001",
            tool="unknown_tool",
            tool_call_args={"weird": "args"},
            policy_decision="denied",
            error="Some error occurred",
        )
        # Should not raise
        result = replay(entry)
        assert result is not None


class TestBatchReplay:
    """Tests for batch replay functionality."""

    def test_batch_replay_empty_list(self):
        """Batch replay with empty list should return empty list."""
        results = batch_replay([])
        assert results == []

    def test_batch_replay_single_entry(self):
        """Batch replay with single entry should return single result."""
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={},
            policy_decision="allowed",
        )
        results = batch_replay([entry])

        assert len(results) == 1
        assert isinstance(results[0], ReplayResult)
        assert results[0].original_entry is entry

    def test_batch_replay_multiple_entries(self):
        """Batch replay should process all entries."""
        entries = [
            LogEntry(
                action_id=f"act_{i:03d}",
                tool="test_tool",
                tool_call_args={"index": i},
                policy_decision="allowed",
            )
            for i in range(5)
        ]
        results = batch_replay(entries)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.original_entry.action_id == f"act_{i:03d}"

    def test_batch_replay_preserves_order(self):
        """Batch replay should preserve entry order."""
        entries = [
            LogEntry(
                action_id="first",
                tool="tool_a",
                tool_call_args={},
                policy_decision="allowed",
            ),
            LogEntry(
                action_id="second",
                tool="tool_b",
                tool_call_args={},
                policy_decision="denied",
            ),
            LogEntry(
                action_id="third",
                tool="tool_c",
                tool_call_args={},
                policy_decision="allowed",
            ),
        ]
        results = batch_replay(entries)

        assert results[0].original_entry.action_id == "first"
        assert results[1].original_entry.action_id == "second"
        assert results[2].original_entry.action_id == "third"

    def test_batch_replay_dry_run_applied_to_all(self):
        """Batch replay should apply dry_run to all entries."""
        entries = [
            LogEntry(
                action_id=f"act_{i}",
                tool="test",
                tool_call_args={},
                policy_decision="allowed",
            )
            for i in range(3)
        ]

        # Test with dry_run=True
        results = batch_replay(entries, dry_run=True)
        assert all(r.dry_run is True for r in results)

        # Test with dry_run=False
        results = batch_replay(entries, dry_run=False)
        assert all(r.dry_run is False for r in results)


class TestReplayResultStructure:
    """Tests for ReplayResult data structure."""

    def test_replay_result_attributes(self):
        """ReplayResult should have all expected attributes."""
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={"location": "Dallas"},
            policy_decision="allowed",
            tool_result={"time": "10:30"},
        )
        result = replay(entry)

        # Check all attributes exist
        assert hasattr(result, "original_entry")
        assert hasattr(result, "replayed_tool_call")
        assert hasattr(result, "new_result")
        assert hasattr(result, "comparison")
        assert hasattr(result, "dry_run")
        assert hasattr(result, "replayed_at")
        assert hasattr(result, "notes")

    def test_replay_result_types(self):
        """ReplayResult attributes should have correct types."""
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={},
            policy_decision="allowed",
        )
        result = replay(entry)

        assert isinstance(result.original_entry, LogEntry)
        assert isinstance(result.replayed_tool_call, ToolCall)
        assert isinstance(result.new_result, dict)
        assert isinstance(result.comparison, dict)
        assert isinstance(result.dry_run, bool)
        assert isinstance(result.replayed_at, datetime)
        assert isinstance(result.notes, str)
