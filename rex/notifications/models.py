"""Notification data model and SQLite-backed store.

``Notification`` holds all metadata needed by the notification engine to
make routing, quiet-hours, and escalation decisions.

``NotificationStore`` persists notifications to ``~/.rex/notifications.db``
using the standard library ``sqlite3`` module — no extra dependencies.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

NotificationPriority = Literal["low", "medium", "high", "critical"]
NotificationChannel = Literal["desktop", "digest", "sms", "email"]

# ---------------------------------------------------------------------------
# Notification model
# ---------------------------------------------------------------------------

_DB_PATH = Path.home() / ".rex" / "notifications.db"


class Notification(BaseModel):
    """A single notification with routing metadata.

    Attributes:
        id: Unique notification identifier (UUID4 by default).
        title: Short heading shown in the notification popup.
        body: Full notification body text.
        source: Origin of the notification (e.g. ``"email"``, ``"task"``,
            ``"system"``).
        priority: Urgency level controlling dispatch behaviour.
        channel: Preferred delivery channel.
        digest_eligible: Whether the notification may be batched into a
            digest instead of dispatched immediately.
        quiet_hours_exempt: When ``True`` the notification bypasses quiet
            hours suppression.
        created_at: UTC datetime when the notification was created.
        delivered_at: UTC datetime when the notification was delivered, or
            ``None`` if not yet delivered.
        read_at: UTC datetime when the user read/dismissed the notification,
            or ``None`` if unread.
        escalation_due_at: UTC datetime after which the notification should
            be escalated to a higher channel, or ``None`` if no escalation
            is scheduled.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    body: str
    source: str
    priority: NotificationPriority = "low"
    channel: NotificationChannel = "desktop"
    digest_eligible: bool = False
    quiet_hours_exempt: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    delivered_at: datetime | None = None
    read_at: datetime | None = None
    escalation_due_at: datetime | None = None


# ---------------------------------------------------------------------------
# NotificationStore
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS notifications (
    id                TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    body              TEXT NOT NULL,
    source            TEXT NOT NULL,
    priority          TEXT NOT NULL,
    channel           TEXT NOT NULL,
    digest_eligible   INTEGER NOT NULL DEFAULT 0,
    quiet_hours_exempt INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL,
    delivered_at      TEXT,
    read_at           TEXT,
    escalation_due_at TEXT
)
"""


def _to_row(n: Notification) -> tuple[object, ...]:
    return (
        n.id,
        n.title,
        n.body,
        n.source,
        n.priority,
        n.channel,
        int(n.digest_eligible),
        int(n.quiet_hours_exempt),
        n.created_at.isoformat(),
        n.delivered_at.isoformat() if n.delivered_at else None,
        n.read_at.isoformat() if n.read_at else None,
        n.escalation_due_at.isoformat() if n.escalation_due_at else None,
    )


def _from_row(row: tuple[object, ...]) -> Notification:
    def _dt(val: object) -> datetime | None:
        if val is None:
            return None
        return datetime.fromisoformat(str(val))

    return Notification(
        id=str(row[0]),
        title=str(row[1]),
        body=str(row[2]),
        source=str(row[3]),
        priority=str(row[4]),  # type: ignore[arg-type]
        channel=str(row[5]),  # type: ignore[arg-type]
        digest_eligible=bool(row[6]),
        quiet_hours_exempt=bool(row[7]),
        created_at=datetime.fromisoformat(str(row[8])),
        delivered_at=_dt(row[9]),
        read_at=_dt(row[10]),
        escalation_due_at=_dt(row[11]),
    )


class NotificationStore:
    """SQLite-backed persistence for :class:`Notification` objects.

    Args:
        db_path: Path to the SQLite database file.  Defaults to
            ``~/.rex/notifications.db``.
    """

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(_CREATE_TABLE)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def add(self, notification: Notification) -> None:
        """Persist *notification* to the store.

        Raises:
            sqlite3.IntegrityError: If a notification with the same ``id``
                already exists.
        """
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO notifications
                    (id, title, body, source, priority, channel,
                     digest_eligible, quiet_hours_exempt, created_at,
                     delivered_at, read_at, escalation_due_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _to_row(notification),
            )

    def upsert(self, notification: Notification) -> None:
        """Insert or replace *notification* in the store.

        Unlike :meth:`add`, this silently overwrites an existing record with
        the same ``id`` rather than raising.
        """
        with self._connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO notifications
                    (id, title, body, source, priority, channel,
                     digest_eligible, quiet_hours_exempt, created_at,
                     delivered_at, read_at, escalation_due_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _to_row(notification),
            )

    def get_unread(self) -> list[Notification]:
        """Return all notifications where ``read_at`` is ``NULL``.

        Results are ordered by ``created_at`` ascending (oldest first).
        """
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM notifications WHERE read_at IS NULL ORDER BY created_at ASC"
            ).fetchall()
        return [_from_row(row) for row in rows]

    def mark_read(self, notification_id: str) -> None:
        """Set ``read_at`` to the current UTC time for *notification_id*.

        Does nothing if the ID does not exist.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as con:
            con.execute(
                "UPDATE notifications SET read_at = ? WHERE id = ?",
                (now, notification_id),
            )

    def update(self, notification: Notification) -> None:
        """Overwrite all mutable fields for the matching ``id``.

        Does nothing if the ID does not exist.
        """
        with self._connect() as con:
            con.execute(
                """
                UPDATE notifications SET
                    title = ?,
                    body = ?,
                    source = ?,
                    priority = ?,
                    channel = ?,
                    digest_eligible = ?,
                    quiet_hours_exempt = ?,
                    created_at = ?,
                    delivered_at = ?,
                    read_at = ?,
                    escalation_due_at = ?
                WHERE id = ?
                """,
                (
                    notification.title,
                    notification.body,
                    notification.source,
                    notification.priority,
                    notification.channel,
                    int(notification.digest_eligible),
                    int(notification.quiet_hours_exempt),
                    notification.created_at.isoformat(),
                    notification.delivered_at.isoformat() if notification.delivered_at else None,
                    notification.read_at.isoformat() if notification.read_at else None,
                    (
                        notification.escalation_due_at.isoformat()
                        if notification.escalation_due_at
                        else None
                    ),
                    notification.id,
                ),
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "Notification",
    "NotificationChannel",
    "NotificationPriority",
    "NotificationStore",
]
