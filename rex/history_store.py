"""Conversation history persistence backed by SQLite.

Stores conversation turns (user / assistant messages) per user so that
sessions can be resumed after restarts.

Default database path: ``data/history.db`` (configurable via ``db_path``).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("data/history.db")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS turns (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   TEXT    NOT NULL,
    role      TEXT    NOT NULL,
    content   TEXT    NOT NULL,
    timestamp TEXT    NOT NULL
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_turns_user_ts ON turns (user_id, timestamp);
"""


class HistoryStore:
    """Thread-safe SQLite-backed store for conversation turns.

    Args:
        db_path: Path to the SQLite database file.  Defaults to
            ``data/history.db``.  Parent directories are created if they
            do not exist.
    """

    def __init__(self, db_path: Path = _DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_SQL)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_turn(
        self,
        user_id: str,
        role: str,
        content: str,
        timestamp: datetime,
    ) -> None:
        """Persist a single conversation turn.

        Args:
            user_id:   Identifier for the user/session.
            role:      ``"user"`` or ``"assistant"``.
            content:   Message text.
            timestamp: When the turn occurred.  Stored as UTC ISO-8601.
        """
        ts = timestamp.astimezone(timezone.utc).isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO turns (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                    (user_id, role, content, ts),
                )

    def load_history(self, user_id: str, limit: int = 50) -> list[dict]:
        """Return the most recent *limit* turns for *user_id*, oldest first.

        Args:
            user_id: Identifier for the user/session.
            limit:   Maximum number of turns to return (default: 50).

        Returns:
            List of dicts with keys ``id``, ``user_id``, ``role``,
            ``content``, ``timestamp``.
        """
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, user_id, role, content, timestamp
                    FROM (
                        SELECT id, user_id, role, content, timestamp
                        FROM turns
                        WHERE user_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    ) sub
                    ORDER BY id ASC
                    """,
                    (user_id, limit),
                )
                return [dict(row) for row in cursor.fetchall()]

    def prune(self, user_id: str, keep_days: int = 30) -> int:
        """Delete turns older than *keep_days* for *user_id*.

        Args:
            user_id:   Identifier for the user/session.
            keep_days: Turns older than this many days are deleted.

        Returns:
            Number of rows deleted.
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
        cutoff_ts = cutoff.isoformat()
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    "DELETE FROM turns WHERE user_id = ? AND timestamp < ?",
                    (user_id, cutoff_ts),
                )
                return cursor.rowcount


__all__ = ["HistoryStore"]
