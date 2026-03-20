"""Local notification dashboard store using SQLite.

Persists notifications delivered via the ``dashboard`` channel so they can
be retrieved by a dashboard UI or API endpoint. The store supports:

- Writing new notifications
- Querying recent/unread/by-priority notifications
- Marking notifications as read
- Automatic retention cleanup (configurable days)

The database is stored locally (default: ``data/dashboard_notifications.db``)
and never exposed to the network on its own.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rex.db_pool import ConnectionPool, ConnectionPoolError, PoolConfig
from rex.notification_priority import NotificationPriority

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("data/dashboard_notifications.db")
_DEFAULT_RETENTION_DAYS = 30


class DashboardStoreConfig(BaseModel):
    """Configuration for the dashboard notification store."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        default="sqlite",
        description="Store type (currently only 'sqlite' is supported)",
    )
    path: str | None = Field(
        default=None,
        description="Database file path (default: data/dashboard_notifications.db)",
    )
    retention_days: int = Field(
        default=30,
        description="Days to retain notifications before automatic cleanup",
    )
    cleanup_schedule: str | None = Field(
        default="interval:86400",
        description=(
            "Scheduler interval for automatic retention cleanup "
            "(e.g. 'interval:86400' for daily). Set to null to disable."
        ),
    )


def load_dashboard_store_config(raw_config: dict[str, Any]) -> DashboardStoreConfig:
    """Parse dashboard store config from the notifications section.

    Args:
        raw_config: Full runtime config dict.

    Returns:
        A ``DashboardStoreConfig``.
    """
    notif_section = raw_config.get("notifications", {})
    if not isinstance(notif_section, dict):
        return DashboardStoreConfig()
    dashboard_section = notif_section.get("dashboard", {})
    if not isinstance(dashboard_section, dict):
        return DashboardStoreConfig()
    store_section = dashboard_section.get("store", {})
    if not isinstance(store_section, dict):
        return DashboardStoreConfig()
    return DashboardStoreConfig.model_validate(store_section)


class DashboardNotification:
    """A notification stored in the dashboard."""

    __slots__ = (
        "id",
        "priority",
        "title",
        "body",
        "channel",
        "timestamp",
        "read",
        "user_id",
        "metadata_json",
    )

    def __init__(
        self,
        *,
        id: str,
        priority: str,
        title: str,
        body: str,
        channel: str = "dashboard",
        timestamp: str,
        read: bool = False,
        user_id: str | None = None,
        metadata_json: str = "{}",
    ) -> None:
        self.id = id
        self.priority = priority
        self.title = title
        self.body = body
        self.channel = channel
        self.timestamp = timestamp
        self.read = read
        self.user_id = user_id
        self.metadata_json = metadata_json

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for API responses."""
        return {
            "id": self.id,
            "priority": self.priority,
            "title": self.title,
            "body": self.body,
            "channel": self.channel,
            "timestamp": self.timestamp,
            "read": self.read,
            "user_id": self.user_id,
            "metadata": json.loads(self.metadata_json) if self.metadata_json else {},
        }


class DashboardStore:
    """SQLite-backed store for dashboard notifications.

    Args:
        db_path: Path to the SQLite database file.
        retention_days: Notifications older than this are purged on cleanup.
        pool_config: Connection pool configuration.  Reads from environment
            variables when not supplied.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        retention_days: int = _DEFAULT_RETENTION_DAYS,
        pool_config: PoolConfig | None = None,
    ) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._retention_days = retention_days
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._pool = ConnectionPool(self._db_path, pool_config)
        self._init_db()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        with self._pool.connect() as conn:
            yield conn

    def _init_db(self) -> None:
        """Create the notifications table if it does not exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY,
                    priority TEXT NOT NULL DEFAULT 'normal',
                    title TEXT NOT NULL,
                    body TEXT NOT NULL DEFAULT '',
                    channel TEXT NOT NULL DEFAULT 'dashboard',
                    timestamp TEXT NOT NULL,
                    read INTEGER NOT NULL DEFAULT 0,
                    user_id TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_notifications_timestamp
                ON notifications (timestamp DESC)
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_notifications_user_id
                ON notifications (user_id)
                """)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(
        self,
        *,
        notification_id: str | None = None,
        priority: NotificationPriority | str = NotificationPriority.MEDIUM,
        title: str,
        body: str,
        channel: str = "dashboard",
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Write a notification to the store.

        Args:
            notification_id: Unique ID (auto-generated if not provided).
            priority: Notification priority.
            title: Notification title.
            body: Notification body.
            channel: Delivery channel.
            user_id: Associated user ID (for user-scoped queries).
            metadata: Additional metadata dict.

        Returns:
            The notification ID.
        """
        nid = notification_id or f"dash_{uuid.uuid4().hex[:16]}"
        ts = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {})
        # Normalise priority: accept enum or string; unknown strings → medium.
        if isinstance(priority, NotificationPriority):
            priority_str = priority.value
        else:
            priority_str = NotificationPriority.from_str(priority).value
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO notifications
                    (id, priority, title, body, channel, timestamp, read, user_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (nid, priority_str, title, body, channel, ts, user_id, meta_json),
            )
        logger.debug("Dashboard notification stored: %s", nid)

        # Best-effort event publish; must never affect write path.
        try:
            from rex.dashboard.sse import NotificationEvent, get_broadcaster

            unread_count = self.count_unread(user_id=user_id)
            get_broadcaster().publish(
                NotificationEvent(
                    type="notification",
                    notification_id=nid,
                    user_id=user_id,
                    unread_count=unread_count,
                )
            )
        except Exception as exc:
            logger.debug("Dashboard SSE publish skipped: %s", exc)

        return nid

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query_recent(
        self,
        *,
        limit: int = 50,
        user_id: str | None = None,
        unread_only: bool = False,
        priority: str | None = None,
    ) -> list[DashboardNotification]:
        """Query recent notifications.

        Args:
            limit: Maximum number to return.
            user_id: Filter to a specific user.
            unread_only: If True, return only unread notifications.
            priority: Filter by priority level.

        Returns:
            List of notifications, newest first.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if unread_only:
            clauses.append("read = 0")
        if priority is not None:
            clauses.append("priority = ?")
            params.append(priority)

        where = ""
        if clauses:
            where = "WHERE " + " AND ".join(clauses)

        sql = f"""
            SELECT id, priority, title, body, channel, timestamp, read, user_id, metadata_json
            FROM notifications
            {where}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            DashboardNotification(
                id=row["id"],
                # Normalise priority on read: legacy values (e.g. "normal")
                # and any unrecognised strings default to "medium".
                priority=NotificationPriority.from_str(row["priority"]).value,
                title=row["title"],
                body=row["body"],
                channel=row["channel"],
                timestamp=row["timestamp"],
                read=bool(row["read"]),
                user_id=row["user_id"],
                metadata_json=row["metadata_json"],
            )
            for row in rows
        ]

    def count_unread(self, *, user_id: str | None = None) -> int:
        """Count unread notifications.

        Args:
            user_id: Scope to a specific user.

        Returns:
            Number of unread notifications.
        """
        if user_id is not None:
            sql = "SELECT COUNT(*) FROM notifications WHERE read = 0 AND user_id = ?"
            params: tuple[Any, ...] = (user_id,)
        else:
            sql = "SELECT COUNT(*) FROM notifications WHERE read = 0"
            params = ()

        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def mark_as_read(self, notification_id: str, *, user_id: str | None = None) -> bool:
        """Mark a notification as read.

        Returns:
            True if the notification was found and updated.
        """
        if user_id is not None:
            sql = "UPDATE notifications SET read = 1 WHERE id = ? AND read = 0 AND user_id = ?"
            params = (notification_id, user_id)
        else:
            sql = "UPDATE notifications SET read = 1 WHERE id = ? AND read = 0"
            params = (notification_id,)  # type: ignore[assignment]

        with self._connect() as conn:
            cursor = conn.execute(sql, params)
        return cursor.rowcount > 0

    def mark_all_read(self, *, user_id: str | None = None) -> int:
        """Mark all notifications as read.

        Args:
            user_id: Scope to a specific user.

        Returns:
            Number of notifications marked as read.
        """
        if user_id is not None:
            sql = "UPDATE notifications SET read = 1 WHERE read = 0 AND user_id = ?"
            params: tuple[Any, ...] = (user_id,)
        else:
            sql = "UPDATE notifications SET read = 1 WHERE read = 0"
            params = ()

        with self._connect() as conn:
            cursor = conn.execute(sql, params)
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup_old(self) -> int:
        """Remove notifications older than the retention period.

        Returns:
            Number of notifications removed.
        """
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=self._retention_days)).isoformat()
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM notifications WHERE timestamp < ?", (cutoff,))
        removed = cursor.rowcount
        if removed:
            logger.info("Dashboard store: removed %d old notifications", removed)
        return removed


# ------------------------------------------------------------------
# Global instance
# ------------------------------------------------------------------

_dashboard_store: DashboardStore | None = None


def get_dashboard_store() -> DashboardStore:
    """Get the global dashboard store instance."""
    global _dashboard_store
    if _dashboard_store is None:
        _dashboard_store = DashboardStore()
    return _dashboard_store


def set_dashboard_store(store: DashboardStore | None) -> None:
    """Set the global dashboard store instance."""
    global _dashboard_store
    _dashboard_store = store


__all__ = [
    "ConnectionPoolError",
    "DashboardNotification",
    "DashboardStore",
    "DashboardStoreConfig",
    "NotificationPriority",
    "PoolConfig",
    "get_dashboard_store",
    "load_dashboard_store_config",
    "set_dashboard_store",
]
