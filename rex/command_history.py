"""Command history store for Rex (US-061).

Records voice/text commands with their result and success status.
Backed by SQLite at ``data/command_history.db``.

Public API
----------
- :class:`CommandHistoryStore` — store class
- :func:`get_history_store`    — singleton accessor
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path("data") / "command_history.db"


def _get_db_path() -> Path:
    raw = os.getenv("REX_DATA_DIR")
    if raw:
        return Path(raw) / "command_history.db"
    return _DEFAULT_DB


class CommandHistoryStore:
    """SQLite-backed command history store.

    Args:
        db_path: Path to the SQLite database file.
                 Defaults to ``data/command_history.db``.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else _get_db_path()
        self._init_db()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._open() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS command_history (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    command   TEXT NOT NULL,
                    result    TEXT,
                    success   INTEGER NOT NULL DEFAULT 1
                )
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        command: str,
        result: str = "",
        success: bool = True,
        timestamp: str | None = None,
    ) -> int:
        """Append a command entry and return its row id.

        Args:
            command:   The command text (voice transcript or typed input).
            result:    The text result returned by Rex.
            success:   Whether the command completed without error.
            timestamp: ISO-8601 UTC timestamp; defaults to *now*.

        Returns:
            The integer row id of the new entry.
        """
        ts = timestamp or datetime.now(UTC).isoformat()
        with self._open() as conn:
            cur = conn.execute(
                "INSERT INTO command_history (timestamp, command, result, success) VALUES (?, ?, ?, ?)",
                (ts, command, result, 1 if success else 0),
            )
            conn.commit()
            return cur.lastrowid or 0

    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent *limit* commands, newest first.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of dicts with keys: ``id``, ``timestamp``, ``command``,
            ``result``, ``success``.
        """
        limit = max(1, min(limit, 500))
        with self._open() as conn:
            rows = conn.execute(
                "SELECT id, timestamp, command, result, success FROM command_history "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "command": row["command"],
                "result": row["result"],
                "success": bool(row["success"]),
            }
            for row in rows
        ]

    def clear(self) -> None:
        """Delete all command history entries."""
        with self._open() as conn:
            conn.execute("DELETE FROM command_history")
            conn.commit()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: CommandHistoryStore | None = None


def get_history_store() -> CommandHistoryStore:
    """Return the global :class:`CommandHistoryStore` singleton."""
    global _store
    if _store is None:
        _store = CommandHistoryStore()
    return _store


__all__ = [
    "CommandHistoryStore",
    "get_history_store",
]
