"""Permissions system for Rex.

Provides a per-user permission model backed by data/users.db.

Permissions
-----------
- ``computer_control`` — read/write/execute files and apps on the desktop
- ``email_send``       — send email via configured email backend
- ``sms_send``         — send SMS via configured messaging backend
- ``ha_control``       — control Home Assistant entities
- ``admin``            — manage users and permissions

The first registered user is automatically granted ``admin``.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

_DB_PATH = Path(os.getenv("REX_DATA_DIR", "data")) / "users.db"


class Permission(str, Enum):
    """Enumeration of all supported permissions."""

    computer_control = "computer_control"
    email_send = "email_send"
    sms_send = "sms_send"
    ha_control = "ha_control"
    admin = "admin"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_db_path() -> Path:
    raw = os.getenv("REX_DATA_DIR")
    if raw:
        return Path(raw) / "users.db"
    return _DB_PATH


def _open_db() -> sqlite3.Connection:
    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create the user_permissions table if it does not already exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_permissions (
            user_id    TEXT NOT NULL,
            permission TEXT NOT NULL,
            PRIMARY KEY (user_id, permission)
        )
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def grant_permission(user_id: str, permission: "Permission | str") -> None:
    """Grant *permission* to *user_id*.

    Silently succeeds if the permission is already granted.

    Args:
        user_id:    UUID of the target user.
        permission: A :class:`Permission` value or its string name.

    Raises:
        ValueError: If *permission* is not a recognised :class:`Permission`.
    """
    perm = Permission(permission) if isinstance(permission, str) else permission
    with _open_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO user_permissions (user_id, permission) VALUES (?, ?)",
            (user_id, perm.value),
        )
        conn.commit()
    logger.info("Granted %s to user %s", perm.value, user_id)


def revoke_permission(user_id: str, permission: "Permission | str") -> None:
    """Revoke *permission* from *user_id*.

    Silently succeeds if the permission was not held.

    Args:
        user_id:    UUID of the target user.
        permission: A :class:`Permission` value or its string name.

    Raises:
        ValueError: If *permission* is not a recognised :class:`Permission`.
    """
    perm = Permission(permission) if isinstance(permission, str) else permission
    with _open_db() as conn:
        conn.execute(
            "DELETE FROM user_permissions WHERE user_id = ? AND permission = ?",
            (user_id, perm.value),
        )
        conn.commit()
    logger.info("Revoked %s from user %s", perm.value, user_id)


def check_permission(user_id: str, permission: "Permission | str") -> bool:
    """Return ``True`` if *user_id* holds *permission*, ``False`` otherwise.

    Args:
        user_id:    UUID of the user to check.
        permission: A :class:`Permission` value or its string name.

    Raises:
        ValueError: If *permission* is not a recognised :class:`Permission`.
    """
    perm = Permission(permission) if isinstance(permission, str) else permission
    with _open_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM user_permissions WHERE user_id = ? AND permission = ?",
            (user_id, perm.value),
        ).fetchone()
    return row is not None


def get_permissions(user_id: str) -> list[str]:
    """Return a list of permission names held by *user_id*.

    Args:
        user_id: UUID of the user.

    Returns:
        List of permission value strings (e.g. ``["admin", "email_send"]``).
    """
    with _open_db() as conn:
        rows = conn.execute(
            "SELECT permission FROM user_permissions WHERE user_id = ?",
            (user_id,),
        ).fetchall()
    return [r["permission"] for r in rows]


def bootstrap_admin_if_first_user(user_id: str) -> None:
    """Grant ``admin`` to *user_id* if no admins exist yet.

    Should be called immediately after a new user is registered so that the
    first account in a fresh deployment automatically becomes the administrator.

    Args:
        user_id: UUID of the newly registered user.
    """
    with _open_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM user_permissions WHERE permission = ?",
            (Permission.admin.value,),
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT OR IGNORE INTO user_permissions (user_id, permission) VALUES (?, ?)",
                (user_id, Permission.admin.value),
            )
            conn.commit()
            logger.info("Bootstrapped admin permission for first user %s", user_id)


__all__ = [
    "Permission",
    "bootstrap_admin_if_first_user",
    "check_permission",
    "get_permissions",
    "grant_permission",
    "revoke_permission",
]
