"""SQLite-backed store for inbound SMS messages received via webhook.

Persists inbound messages so they can be retrieved later via the CLI
(``rex msg receive``) or the messaging service API.  Follows the same
patterns as ``rex.dashboard_store`` (SQLite, config-driven path, global
instance accessor).

The store never logs message bodies or phone numbers at INFO level to
avoid leaking PII in production logs.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("data/inbound_sms.db")
_DEFAULT_RETENTION_DAYS = 90


class InboundStoreConfig(BaseModel):
    """Configuration for the inbound SMS store."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Enable the inbound SMS webhook and store",
    )
    store_path: str | None = Field(
        default=None,
        description="SQLite database file path (default: data/inbound_sms.db)",
    )
    retention_days: int = Field(
        default=90,
        description="Days to retain inbound messages before automatic cleanup",
    )
    auth_token_ref: str = Field(
        default="twilio:inbound",
        description="Credential ref for the Twilio auth token used for signature verification",
    )
    rate_limit: str = Field(
        default="120 per minute",
        description="Rate limit for the inbound webhook endpoint (Flask-Limiter format)",
    )
    cleanup_schedule: str | None = Field(
        default="interval:86400",
        description=(
            "Scheduler interval for automatic retention cleanup "
            "(e.g. 'interval:86400' for daily).  Only active when enabled=true. "
            "Set to null to disable."
        ),
    )


def load_inbound_store_config(raw_config: dict[str, Any]) -> InboundStoreConfig:
    """Parse inbound store config from the messaging section.

    Args:
        raw_config: Full runtime config dict.

    Returns:
        An ``InboundStoreConfig``.
    """
    messaging_section = raw_config.get("messaging", {})
    if not isinstance(messaging_section, dict):
        return InboundStoreConfig()
    inbound_section = messaging_section.get("inbound", {})
    if not isinstance(inbound_section, dict):
        return InboundStoreConfig()
    return InboundStoreConfig.model_validate(inbound_section)


class InboundSmsRecord:
    """A single inbound SMS record stored in the database."""

    __slots__ = (
        "id",
        "sid",
        "from_number",
        "to_number",
        "body",
        "received_at",
        "account_id",
        "user_id",
        "routed",
    )

    def __init__(
        self,
        *,
        id: str | None = None,
        sid: str = "",
        from_number: str = "",
        to_number: str = "",
        body: str = "",
        received_at: datetime | None = None,
        account_id: str | None = None,
        user_id: str | None = None,
        routed: bool = False,
    ) -> None:
        self.id = id or f"inb_{uuid.uuid4().hex[:16]}"
        self.sid = sid
        self.from_number = from_number
        self.to_number = to_number
        self.body = body
        self.received_at = received_at or datetime.now(timezone.utc)
        self.account_id = account_id
        self.user_id = user_id
        self.routed = routed

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "id": self.id,
            "sid": self.sid,
            "from_number": self.from_number,
            "to_number": self.to_number,
            "body": self.body,
            "received_at": self.received_at.isoformat() if self.received_at else None,
            "account_id": self.account_id,
            "user_id": self.user_id,
            "routed": self.routed,
        }


_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS inbound_sms (
    id            TEXT PRIMARY KEY,
    sid           TEXT NOT NULL DEFAULT '',
    from_number   TEXT NOT NULL DEFAULT '',
    to_number     TEXT NOT NULL DEFAULT '',
    body          TEXT NOT NULL DEFAULT '',
    received_at   TEXT NOT NULL,
    account_id    TEXT,
    user_id       TEXT,
    routed        INTEGER NOT NULL DEFAULT 0
);
"""

_CREATE_INDEX_SQL = """\
CREATE INDEX IF NOT EXISTS idx_inbound_sms_received
    ON inbound_sms (received_at DESC);
"""


class InboundSmsStore:
    """SQLite-backed store for inbound SMS messages."""

    def __init__(self, db_path: Path | None = None, retention_days: int = _DEFAULT_RETENTION_DAYS):
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._retention_days = retention_days
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create the table and indexes if they do not exist.

        Also runs lightweight migrations for columns added after the
        initial schema (e.g. ``user_id``).  Each migration is idempotent
        and safe to re-run.
        """
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_SQL)
            self._migrate_add_user_id(conn)

    @staticmethod
    def _migrate_add_user_id(conn: sqlite3.Connection) -> None:
        """Add ``user_id`` column if the table was created before it existed.

        Uses ``PRAGMA table_info`` to detect the column; the ALTER TABLE
        is only executed when the column is absent.  This keeps the
        migration idempotent and safe for concurrent opens.
        """
        columns = {row[1] for row in conn.execute("PRAGMA table_info(inbound_sms)").fetchall()}
        if "user_id" not in columns:
            conn.execute("ALTER TABLE inbound_sms ADD COLUMN user_id TEXT")
            logger.info("Migrated inbound_sms: added user_id column")

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for a database connection."""
        conn = sqlite3.connect(str(self._db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def write(self, record: InboundSmsRecord) -> str:
        """Persist an inbound SMS record.

        Args:
            record: The inbound SMS record to store.

        Returns:
            The record ID.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO inbound_sms
                    (id, sid, from_number, to_number, body, received_at,
                     account_id, user_id, routed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.sid,
                    record.from_number,
                    record.to_number,
                    record.body,
                    record.received_at.isoformat() if record.received_at else "",
                    record.account_id,
                    record.user_id,
                    1 if record.routed else 0,
                ),
            )
        logger.debug("Stored inbound SMS record id=%s", record.id)
        return record.id

    def query_recent(
        self,
        *,
        limit: int = 20,
        user_id: str | None = None,
        account_id: str | None = None,
    ) -> list[InboundSmsRecord]:
        """Query recent inbound messages.

        Args:
            limit: Maximum records to return.
            user_id: Filter by user ID (optional).
            account_id: Filter by account ID (optional).

        Returns:
            List of records, newest first.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if account_id is not None:
            clauses.append("account_id = ?")
            params.append(account_id)

        where = ""
        if clauses:
            where = "WHERE " + " AND ".join(clauses)

        sql = f"""
            SELECT id, sid, from_number, to_number, body, received_at,
                   account_id, user_id, routed
            FROM inbound_sms
            {where}
            ORDER BY received_at DESC
            LIMIT ?
        """
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_record(row) for row in rows]

    def count(self, *, user_id: str | None = None) -> int:
        """Count stored inbound messages.

        Args:
            user_id: Filter by user ID (optional).

        Returns:
            Number of matching records.
        """
        if user_id:
            sql = "SELECT COUNT(*) FROM inbound_sms WHERE user_id = ?"
            params: tuple[Any, ...] = (user_id,)
        else:
            sql = "SELECT COUNT(*) FROM inbound_sms"
            params = ()

        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return row[0] if row else 0

    def cleanup_old(self) -> int:
        """Remove records older than retention_days.

        Returns:
            Number of records deleted.
        """
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=self._retention_days)).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM inbound_sms WHERE received_at < ?",
                (cutoff,),
            )
            deleted = cursor.rowcount
        if deleted:
            logger.info("Cleaned up %d old inbound SMS records", deleted)
        return deleted

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> InboundSmsRecord:
        """Convert a database row to an InboundSmsRecord."""
        received_at_str = row["received_at"]
        try:
            received_at = datetime.fromisoformat(received_at_str)
            if received_at.tzinfo is None:
                received_at = received_at.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            received_at = datetime.now(timezone.utc)

        return InboundSmsRecord(
            id=row["id"],
            sid=row["sid"],
            from_number=row["from_number"],
            to_number=row["to_number"],
            body=row["body"],
            received_at=received_at,
            account_id=row["account_id"],
            user_id=row["user_id"],
            routed=bool(row["routed"]),
        )


# --- Global instance ---

_inbound_store: InboundSmsStore | None = None


def get_inbound_store() -> InboundSmsStore | None:
    """Get the global inbound SMS store instance (or None if not initialized)."""
    return _inbound_store


def set_inbound_store(store: InboundSmsStore | None) -> None:
    """Set the global inbound SMS store instance."""
    global _inbound_store
    _inbound_store = store


def init_inbound_store(raw_config: dict[str, Any] | None = None) -> InboundSmsStore | None:
    """Initialize the global inbound store from config.

    Args:
        raw_config: Full runtime config dict.

    Returns:
        The store instance, or None if inbound is not enabled.
    """
    global _inbound_store

    if raw_config is None:
        return None

    config = load_inbound_store_config(raw_config)
    if not config.enabled:
        logger.debug("Inbound SMS store is disabled in config")
        return None

    db_path = Path(config.store_path) if config.store_path else _DEFAULT_DB_PATH
    _inbound_store = InboundSmsStore(
        db_path=db_path,
        retention_days=config.retention_days,
    )
    logger.info("Inbound SMS store initialized at %s", db_path)
    return _inbound_store


__all__ = [
    "InboundSmsRecord",
    "InboundSmsStore",
    "InboundStoreConfig",
    "get_inbound_store",
    "init_inbound_store",
    "load_inbound_store_config",
    "set_inbound_store",
]
