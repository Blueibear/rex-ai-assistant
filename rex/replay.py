"""Replay mechanism for audit log entries.

This module provides functionality to replay past tool executions from
the audit log. This is useful for:

- Debugging: Reproduce past actions to understand behavior
- Testing: Verify that tools produce consistent results
- Analysis: Compare current behavior against historical results

IMPORTANT: The current implementation is a stub that does NOT actually
execute tools. It reconstructs the ToolCall and returns a placeholder
result. Full replay integration with actual tool execution (in dry-run
mode) will be implemented in a future phase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from rex.audit import LogEntry
from rex.contracts import ToolCall

logger = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    """Result from replaying an audit log entry.

    Attributes:
        original_entry: The original audit log entry being replayed.
        replayed_tool_call: The reconstructed ToolCall from the log entry.
        new_result: The result from replaying (placeholder in stub).
        comparison: Comparison between original and new results.
        dry_run: Whether this was a dry-run (no side effects).
        replayed_at: Timestamp when the replay was performed.
        notes: Additional notes or warnings about the replay.
    """

    original_entry: LogEntry
    replayed_tool_call: ToolCall
    new_result: dict[str, Any] | None
    comparison: dict[str, Any]
    dry_run: bool
    replayed_at: datetime
    notes: str


def reconstruct_tool_call(entry: LogEntry) -> ToolCall:
    """Reconstruct a ToolCall from an audit log entry.

    Args:
        entry: The audit log entry containing tool information.

    Returns:
        A ToolCall instance with the original tool and arguments.
    """
    return ToolCall(
        tool=entry.tool,
        args=entry.tool_call_args,
        requested_by=f"replay:{entry.action_id}",
        idempotency_key=f"replay-{entry.action_id}",
        created_at=datetime.now(timezone.utc),
    )


def replay(entry: LogEntry, *, dry_run: bool = True) -> ReplayResult:
    """Replay a tool execution from an audit log entry.

    This function reconstructs the original tool call from the audit log
    and attempts to re-execute it. By default, it operates in dry-run mode
    which prevents any side effects.

    STUB IMPLEMENTATION: This is currently a stub that does NOT actually
    execute the tool. It returns a placeholder result to demonstrate the
    interface. Full implementation will come in a future phase.

    Args:
        entry: The audit log entry to replay.
        dry_run: If True (default), do not commit any side effects.
            Note: Even with dry_run=False, the current stub does not
            execute actual tools.

    Returns:
        A ReplayResult containing the reconstructed tool call and
        placeholder result.

    Example:
        >>> from rex.audit import AuditLogger, LogEntry
        >>> entry = LogEntry(
        ...     action_id="act_001",
        ...     tool="time_now",
        ...     tool_call_args={"location": "Dallas, TX"},
        ...     policy_decision="allowed",
        ... )
        >>> result = replay(entry)
        >>> result.notes
        'STUB: Tool execution not implemented...'
    """
    logger.info(
        "Replaying action_id=%s, tool=%s (dry_run=%s)",
        entry.action_id,
        entry.tool,
        dry_run,
    )

    # Reconstruct the tool call from the log entry
    tool_call = reconstruct_tool_call(entry)

    # STUB: In full implementation, this would:
    # 1. Get the tool executor from the registry
    # 2. Execute in dry-run mode (no side effects)
    # 3. Compare the new result with the original
    #
    # For now, we return a placeholder result
    stub_result: dict[str, Any] = {
        "status": "stub",
        "message": "Replay execution not yet implemented",
        "original_tool": entry.tool,
        "original_args": entry.tool_call_args,
        "would_execute": not dry_run,
    }

    # Create comparison (placeholder for now)
    comparison: dict[str, Any] = {
        "identical": False,
        "reason": "Cannot compare - stub implementation",
        "original_result": entry.tool_result,
        "new_result": stub_result,
    }

    notes = (
        "STUB: Tool execution not implemented. "
        "Full replay integration will be added in a future phase. "
        "The reconstructed ToolCall is available for inspection."
    )

    return ReplayResult(
        original_entry=entry,
        replayed_tool_call=tool_call,
        new_result=stub_result,
        comparison=comparison,
        dry_run=dry_run,
        replayed_at=datetime.now(timezone.utc),
        notes=notes,
    )


def batch_replay(
    entries: list[LogEntry],
    *,
    dry_run: bool = True,
) -> list[ReplayResult]:
    """Replay multiple audit log entries.

    Args:
        entries: List of audit log entries to replay.
        dry_run: If True (default), do not commit any side effects.

    Returns:
        List of ReplayResult objects for each entry.
    """
    results = []
    for entry in entries:
        try:
            result = replay(entry, dry_run=dry_run)
            results.append(result)
        except Exception as e:
            logger.error(
                "Failed to replay action_id=%s: %s",
                entry.action_id,
                e,
            )
            # Create a failure result
            results.append(
                ReplayResult(
                    original_entry=entry,
                    replayed_tool_call=reconstruct_tool_call(entry),
                    new_result=None,
                    comparison={"error": str(e)},
                    dry_run=dry_run,
                    replayed_at=datetime.now(timezone.utc),
                    notes=f"Replay failed: {e}",
                )
            )
    return results


__all__ = [
    "ReplayResult",
    "reconstruct_tool_call",
    "replay",
    "batch_replay",
]
