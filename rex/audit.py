"""Audit logging for Rex tool execution.

This module provides audit logging functionality for tracking all tool
invocations, policy decisions, and execution results. It supports:

- Persistent storage in JSON Lines format
- Automatic redaction of sensitive fields
- Time-range queries for log retrieval
- Export functionality for archival

All log entries are automatically redacted using redact_sensitive_keys()
before being written to disk to prevent sensitive data exposure.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from pydantic import BaseModel, Field

from rex.contracts.core import redact_sensitive_keys

logger = logging.getLogger(__name__)

# Default log directory path
DEFAULT_LOG_DIR = Path("data/logs")
DEFAULT_LOG_FILE = "audit.log"


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


class LogEntry(BaseModel):
    """A single audit log entry for tool execution.

    Each entry captures the full context of a tool invocation including
    the tool name, arguments, policy decision, and result. Sensitive
    fields are redacted before persistence.
    """

    timestamp: datetime = Field(
        default_factory=_utc_now,
        description="When this log entry was created (UTC)",
    )
    action_id: str = Field(
        ...,
        description="Unique identifier for this action",
    )
    task_id: str | None = Field(
        default=None,
        description="ID of the parent task, if any",
    )
    tool: str = Field(
        ...,
        description="Name of the tool that was invoked",
    )
    tool_call_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the tool (redacted before storage)",
    )
    policy_decision: Literal["allowed", "denied", "requires_approval"] = Field(
        ...,
        description="Policy engine decision for this tool call",
    )
    tool_result: dict[str, Any] | None = Field(
        default=None,
        description="Result from tool execution (redacted before storage)",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the tool execution failed",
    )
    redacted: bool = Field(
        default=False,
        description="Whether sensitive fields have been redacted",
    )
    requested_by: str | None = Field(
        default=None,
        description="Identifier of who/what requested this action",
    )
    duration_ms: int | None = Field(
        default=None,
        description="Execution duration in milliseconds",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "timestamp": "2024-01-15T10:30:00Z",
                    "action_id": "act_001",
                    "task_id": "task_001",
                    "tool": "time_now",
                    "tool_call_args": {"location": "Dallas, TX"},
                    "policy_decision": "allowed",
                    "tool_result": {"local_time": "2024-01-15 10:30"},
                    "redacted": True,
                    "duration_ms": 15,
                }
            ]
        }
    }


class AuditLogger:
    """Persistent audit logger for tool executions.

    Stores audit entries in JSON Lines format with automatic redaction
    of sensitive fields. Thread-safe for concurrent writes.

    Usage:
        logger = AuditLogger()
        entry = LogEntry(
            action_id="act_001",
            tool="time_now",
            tool_call_args={"location": "Dallas"},
            policy_decision="allowed",
        )
        logger.log(entry)
    """

    def __init__(
        self,
        log_dir: Path | str | None = None,
        log_file: str = DEFAULT_LOG_FILE,
        log_path: Path | str | None = None,
    ) -> None:
        """Initialize the audit logger.

        Args:
            log_dir: Directory to store log files. Defaults to data/logs.
            log_file: Name of the log file. Defaults to audit.log.
            log_path: Optional full path to the log file. Overrides log_dir/log_file.
        """
        if log_path is not None:
            self._log_path = Path(log_path)
            self._log_dir = self._log_path.parent
            self._log_file = self._log_path.name
        else:
            if log_dir is None:
                log_dir = DEFAULT_LOG_DIR
            self._log_dir = Path(log_dir)
            self._log_file = log_file
            self._log_path = self._log_dir / self._log_file
        self._lock = Lock()
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        """Create the log directory if it doesn't exist."""
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("Could not create log directory %s: %s", self._log_dir, e)

    @property
    def log_path(self) -> Path:
        """Return the path to the audit log file."""
        return self._log_path

    def _redact_entry(self, entry: LogEntry) -> LogEntry:
        """Apply redaction to sensitive fields in a log entry.

        Args:
            entry: The log entry to redact.

        Returns:
            A new LogEntry with sensitive fields redacted.
        """
        redacted_args = redact_sensitive_keys(entry.tool_call_args)
        redacted_result = (
            redact_sensitive_keys(entry.tool_result) if entry.tool_result is not None else None
        )

        return LogEntry(
            timestamp=entry.timestamp,
            action_id=entry.action_id,
            task_id=entry.task_id,
            tool=entry.tool,
            tool_call_args=redacted_args,
            policy_decision=entry.policy_decision,
            tool_result=redacted_result,
            error=entry.error,
            redacted=True,
            requested_by=entry.requested_by,
            duration_ms=entry.duration_ms,
        )

    def log(self, entry: LogEntry) -> None:
        """Log an audit entry to the persistent store.

        Redacts sensitive fields before writing to disk.

        Args:
            entry: The log entry to persist.
        """
        redacted_entry = self._redact_entry(entry)

        # Serialize with ISO format for datetime
        json_line = redacted_entry.model_dump_json() + "\n"

        with self._lock:
            try:
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(json_line)
            except OSError as e:
                logger.error("Failed to write audit log: %s", e)

    def read(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[LogEntry]:
        """Read log entries from the persistent store.

        Args:
            start_time: Optional start of time range (inclusive).
            end_time: Optional end of time range (inclusive).

        Returns:
            List of log entries matching the time range criteria.
        """
        entries: list[LogEntry] = []

        if not self._log_path.exists():
            return entries

        try:
            with open(self._log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = LogEntry.model_validate_json(line)
                        # Apply time range filter
                        if start_time is not None and entry.timestamp < start_time:
                            continue
                        if end_time is not None and entry.timestamp > end_time:
                            continue
                        entries.append(entry)
                    except Exception as e:
                        logger.warning("Skipping malformed log entry: %s", e)
        except OSError as e:
            logger.error("Failed to read audit log: %s", e)

        return entries

    def read_by_action_id(self, action_id: str) -> LogEntry | None:
        """Read a specific log entry by action ID.

        Args:
            action_id: The action ID to look up.

        Returns:
            The matching log entry or None if not found.
        """
        if not self._log_path.exists():
            return None

        try:
            with open(self._log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = LogEntry.model_validate_json(line)
                        if entry.action_id == action_id:
                            return entry
                    except Exception:
                        continue
        except OSError as e:
            logger.error("Failed to read audit log: %s", e)

        return None

    def export(self, path: str | Path) -> int:
        """Export all log entries to a specified path.

        Args:
            path: Destination path for the export file.

        Returns:
            Number of entries exported.
        """
        entries = self.read()
        export_path = Path(path)

        # Ensure parent directory exists
        export_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(export_path, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(entry.model_dump_json() + "\n")
            logger.info("Exported %d audit entries to %s", len(entries), export_path)
            return len(entries)
        except OSError as e:
            logger.error("Failed to export audit log: %s", e)
            return 0

    def clear(self) -> None:
        """Clear all entries from the audit log.

        WARNING: This permanently deletes all audit history.
        Use only for testing or with explicit user confirmation.
        """
        with self._lock:
            try:
                if self._log_path.exists():
                    self._log_path.unlink()
            except OSError as e:
                logger.error("Failed to clear audit log: %s", e)


# Module-level singleton
_default_audit_logger: AuditLogger | None = None
_singleton_lock = Lock()


def get_audit_logger() -> AuditLogger:
    """Get the default audit logger singleton.

    Returns:
        The shared AuditLogger instance.
    """
    global _default_audit_logger
    with _singleton_lock:
        if _default_audit_logger is None:
            _default_audit_logger = AuditLogger()
        return _default_audit_logger


def reset_audit_logger() -> None:
    """Reset the default audit logger singleton.

    Primarily used for testing to ensure a clean state.
    """
    global _default_audit_logger
    with _singleton_lock:
        _default_audit_logger = None


def replay(entry: LogEntry, *, dry_run: bool = True) -> Any:
    """Replay a tool execution from an audit log entry.

    This is a convenience wrapper for rex.replay.replay.
    """
    from rex.replay import replay as _replay

    return _replay(entry, dry_run=dry_run)


__all__ = [
    "LogEntry",
    "AuditLogger",
    "get_audit_logger",
    "reset_audit_logger",
    "replay",
    "DEFAULT_LOG_DIR",
    "DEFAULT_LOG_FILE",
]
