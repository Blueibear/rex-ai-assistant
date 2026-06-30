"""Tests for Rex replay functionality."""

from __future__ import annotations

from datetime import UTC, datetime

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
        old_time = datetime(2020, 1, 1, tzinfo=UTC)
        entry = LogEntry(
            timestamp=old_time,
            action_id="act_004",
            tool="test_tool",
            tool_call_args={},
            policy_decision="allowed",
        )
        tool_call = reconstruct_tool_call(entry)

        assert tool_call.created_at > old_time


class TestReplay:
    """Tests for the replay function."""

    def test_replay_raises_not_implemented(self):
        """replay() must raise NotImplementedError — never return a result dict."""
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={"location": "Dallas"},
            policy_decision="allowed",
        )
        with pytest.raises(NotImplementedError, match="replay is not available in this build"):
            replay(entry)

    def test_replay_raises_regardless_of_dry_run(self):
        """replay() raises whether dry_run is True or False."""
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={},
            policy_decision="allowed",
        )
        with pytest.raises(NotImplementedError):
            replay(entry, dry_run=True)

        with pytest.raises(NotImplementedError):
            replay(entry, dry_run=False)

    def test_replay_never_returns_placeholder_dict(self):
        """replay() must not return a dict with placeholder or stub content."""
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={"location": "Dallas"},
            policy_decision="allowed",
        )
        try:
            result = replay(entry)
            # If it somehow returned, verify it's not a placeholder
            assert result is not None
            if isinstance(result, ReplayResult) and result.new_result is not None:
                assert "stub" not in str(result.new_result).lower()
        except NotImplementedError:
            pass  # Expected path

    def test_audit_replay_wrapper_raises_not_implemented(self):
        """The rex.audit.replay convenience wrapper also raises NotImplementedError."""
        entry = LogEntry(
            action_id="act_006",
            tool="time_now",
            tool_call_args={"location": "Dallas"},
            policy_decision="allowed",
        )
        with pytest.raises(NotImplementedError, match="replay is not available in this build"):
            audit_replay(entry)


class TestBatchReplay:
    """Tests for batch replay functionality."""

    def test_batch_replay_empty_list(self):
        """Batch replay with empty list should return empty list."""
        results = batch_replay([])
        assert results == []

    def test_batch_replay_single_entry(self):
        """Batch replay with single entry returns one failure result."""
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
        """Batch replay should apply dry_run to all failure results."""
        entries = [
            LogEntry(
                action_id=f"act_{i}",
                tool="test",
                tool_call_args={},
                policy_decision="allowed",
            )
            for i in range(3)
        ]

        results = batch_replay(entries, dry_run=True)
        assert all(r.dry_run is True for r in results)

        results = batch_replay(entries, dry_run=False)
        assert all(r.dry_run is False for r in results)

    def test_batch_replay_failure_new_result_is_none(self):
        """Batch replay failure results must have new_result=None, not a dict."""
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={},
            policy_decision="allowed",
        )
        results = batch_replay([entry])

        assert len(results) == 1
        assert results[0].new_result is None

    def test_batch_replay_failure_notes_contain_error(self):
        """Batch replay failure notes should contain the error message."""
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={},
            policy_decision="allowed",
        )
        results = batch_replay([entry])

        assert len(results) == 1
        assert "replay is not available in this build" in results[0].notes


class TestReplayResultStructure:
    """Tests for ReplayResult data structure."""

    def test_replay_result_can_be_instantiated(self):
        """ReplayResult dataclass should be directly instantiable."""
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={"location": "Dallas"},
            policy_decision="allowed",
        )
        tool_call = reconstruct_tool_call(entry)
        now = datetime.now(UTC)

        result = ReplayResult(
            original_entry=entry,
            replayed_tool_call=tool_call,
            new_result=None,
            comparison={"error": "replay is not available in this build"},
            dry_run=True,
            replayed_at=now,
            notes="Replay failed: replay is not available in this build",
        )

        assert isinstance(result, ReplayResult)
        assert result.original_entry is entry
        assert result.replayed_tool_call is tool_call
        assert result.new_result is None
        assert isinstance(result.comparison, dict)
        assert result.dry_run is True
        assert result.replayed_at is now
        assert isinstance(result.notes, str)

    def test_replay_result_attributes_exist(self):
        """ReplayResult should have all expected attributes."""
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={},
            policy_decision="allowed",
        )
        tool_call = reconstruct_tool_call(entry)
        result = ReplayResult(
            original_entry=entry,
            replayed_tool_call=tool_call,
            new_result=None,
            comparison={},
            dry_run=True,
            replayed_at=datetime.now(UTC),
            notes="",
        )

        assert hasattr(result, "original_entry")
        assert hasattr(result, "replayed_tool_call")
        assert hasattr(result, "new_result")
        assert hasattr(result, "comparison")
        assert hasattr(result, "dry_run")
        assert hasattr(result, "replayed_at")
        assert hasattr(result, "notes")
