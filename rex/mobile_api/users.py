"""Mobile user lookup, credential verification, and display projection.

All reads go against the canonical ``data/users.db`` tables (``users``,
``user_permissions``).  User IDs are authorization keys: every path validates
them with :func:`rex.identity.validate_user_id` before any private access.

Passwords, hashes, and tokens are never logged from this module.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

import bcrypt

from rex.identity import get_user_profile, validate_user_id
from rex.mobile_api.db import connect

logger = logging.getLogger(__name__)

# Presentation-only defaults.  `role` and `color` are display projections;
# authorization always uses the live permission set.
DEFAULT_USER_COLOR = "#2D7FF9"

# A real bcrypt hash of an unguessable throwaway value, used to equalise
# timing between "unknown username" and "wrong password" login failures.
_DUMMY_BCRYPT_HASH = bcrypt.hashpw(b"askrex-dummy-timing-equalizer", bcrypt.gensalt())


def find_user_by_username(conn: sqlite3.Connection, username: str) -> sqlite3.Row | None:
    """Return the user row for *username*, or None."""
    row: sqlite3.Row | None = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    return row


def get_user(conn: sqlite3.Connection, user_id: str) -> sqlite3.Row | None:
    """Return the user row for a validated *user_id*, or None.

    Raises:
        ValueError: If *user_id* fails canonical validation (fails closed
            before any database read).
    """
    user_id = validate_user_id(user_id)
    row: sqlite3.Row | None = conn.execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    return row


def is_user_active(row: sqlite3.Row | None) -> bool:
    """Return True when the user row exists and is not disabled."""
    if row is None:
        return False
    try:
        disabled_at = row["disabled_at"]
    except (IndexError, KeyError):
        # Pre-migration row shape: no disabled_at column means active.
        disabled_at = None
    return disabled_at is None


def verify_user_credentials(db_path: Any, username: str, password: str) -> sqlite3.Row | None:
    """Verify username/password against the canonical users table.

    Returns the user row only when the password matches AND the user is
    active AND the stored user ID is canonically valid.  Every failure mode
    returns None so callers produce one non-enumerating error.
    """
    username = (username or "").strip()
    if not username or not password:
        return None
    conn = connect(db_path)
    try:
        row = find_user_by_username(conn, username)
        if row is None:
            # Equalise timing with the real-password path.
            bcrypt.checkpw(password.encode(), _DUMMY_BCRYPT_HASH)
            return None
        if not bcrypt.checkpw(password.encode(), row["password"].encode()):
            return None
        if not is_user_active(row):
            logger.info("Mobile login rejected for disabled user account")
            return None
        try:
            validate_user_id(row["id"])
        except ValueError:
            logger.warning("Mobile login rejected: stored user ID failed validation")
            return None
        return row
    finally:
        conn.close()


def get_user_permissions(db_path: Any, user_id: str) -> list[str]:
    """Return the live permission names for a validated *user_id*."""
    user_id = validate_user_id(user_id)
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT permission FROM user_permissions WHERE user_id = ?",
            (user_id,),
        ).fetchall()
    finally:
        conn.close()
    return sorted(row["permission"] for row in rows)


def role_projection(permissions: list[str] | frozenset[str]) -> str:
    """Map the live permission set to the presentation-only role string."""
    return "owner" if "admin" in permissions else "member"


def build_user_projection(db_path: Any, user_id: str, username: str) -> dict[str, Any]:
    """Return the live user display projection for API responses.

    ``role`` and ``color`` are presentation data only; the server never
    authorizes from them.
    """
    user_id = validate_user_id(user_id)
    permissions = get_user_permissions(db_path, user_id)
    display_name = username
    profile = get_user_profile(user_id)
    if profile and isinstance(profile.get("name"), str) and profile["name"].strip():
        display_name = profile["name"].strip()
    return {
        "id": user_id,
        "name": display_name,
        "role": role_projection(permissions),
        "permissions": permissions,
        "color": DEFAULT_USER_COLOR,
    }


__all__ = [
    "DEFAULT_USER_COLOR",
    "build_user_projection",
    "find_user_by_username",
    "get_user",
    "get_user_permissions",
    "is_user_active",
    "role_projection",
    "verify_user_credentials",
]
