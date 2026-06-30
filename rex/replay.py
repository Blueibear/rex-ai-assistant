"""Replay mechanism for audit log entries.

This module provides utilities for reconstructing tool calls from the
audit log. Replay execution is not available in this build: calling
``replay()`` raises ``NotImplementedError``. The ``reconstruct_tool_call()``
utility and ``ReplayResult`` dataclass remain available for inspection and
testing purposes.

Note: ``get_replayable_calls`` (read-only audit log query) lives in
``rex.audit`` and is unaffected by this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from rex.audit import LogEntry
from rex.contracts import ToolCall

logger = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    """Result from a replay attempt.

    Attributes:
        original_entry: The original audit log entry being replayed.
        replayed_tool_call: The reconstructed ToolCall from the log entry.
        new_result: The result from replaying, or None on failure.
        comparison: Comparison between original and new results.
        dry_run: Whether this was a dry-run (no side effects).
        replayed_at: Timestamp when the replay was attempted.
        notes: Additional notes or error details about the replay attempt.
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
        created_at=datetime.now(UTC),
    )


def replay(entry: LogEntry, *, dry_run: bool = True) -> ReplayResult:
    """Replay a tool execution from an audit log entry.

    Raises:
        NotImplementedError: Replay execution is not available in this build.

    Args:
        entry: The audit log entry to replay.
        dry_run: Reserved for future use.
    """
    raise NotImplementedError("replay is not available in this build")


def batch_replay(
    entries: list[LogEntry],
    *,
    dry_run: bool = True,
) -> list[ReplayResult]:
    """Attempt to replay multiple audit log entries.

    Each entry is attempted independently. On failure, a ``ReplayResult`` with
    ``new_result=None`` and the error in ``notes`` is returned for that entry.

    Args:
        entries: List of audit log entries to replay.
        dry_run: Reserved for future use.

    Returns:
        List of ReplayResult objects — one per entry. On failure, ``new_result``
        is ``None`` and ``notes`` contains the error message.
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
            results.append(
                ReplayResult(
                    original_entry=entry,
                    replayed_tool_call=reconstruct_tool_call(entry),
                    new_result=None,
                    comparison={"error": str(e)},
                    dry_run=dry_run,
                    replayed_at=datetime.now(UTC),
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
