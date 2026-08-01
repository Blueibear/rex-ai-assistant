"""Canonical users database access and idempotent mobile schema migrations.

The mobile gateway reuses ``data/users.db`` (the database owned by
``rex.auth`` / ``rex.permissions``).  It never creates a second users
database.  This module adds, through idempotent migrations:

- ``users.disabled_at`` — explicit user-active state (existing rows stay
  active because the new column defaults to NULL);
- ``mobile_sessions`` — per-device mobile sessions;
- ``mobile_refresh_tokens`` — hashed, rotating refresh-token families.

Raw refresh tokens are never stored; only SHA-256 hashes.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from rex.runtime_paths import household_data_path

logger = logging.getLogger(__name__)

_BUSY_TIMEOUT_MS = 5000


def default_users_db_path() -> Path:
    """Return the canonical household users database path."""
    return household_data_path("users.db")


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open a connection to the users database with safe defaults.

    ``isolation_level=None`` puts sqlite3 in autocommit mode so transaction
    boundaries are explicit (``BEGIN IMMEDIATE`` / ``COMMIT``), which the
    refresh-rotation race guarantees depend on.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row["name"] for row in rows}


def migrate_users_db(db_path: Path | str) -> None:
    """Apply all mobile gateway migrations to the users database.

    Safe to run repeatedly and safe on fresh, legacy, and already-migrated
    databases.  Existing users, password hashes, and permissions are
    preserved; existing users remain active (``disabled_at`` defaults NULL).
    """
    conn = connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        # Canonical tables (identical shape to rex.auth / rex.permissions) so
        # a fresh database is usable without importing those modules first.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id       TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created  TEXT NOT NULL
            )
            """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_permissions (
                user_id    TEXT NOT NULL,
                permission TEXT NOT NULL,
                PRIMARY KEY (user_id, permission)
            )
            """)
        if "disabled_at" not in _table_columns(conn, "users"):
            conn.execute("ALTER TABLE users ADD COLUMN disabled_at TEXT NULL")
            logger.info("users.db migration: added users.disabled_at column")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mobile_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                device_name TEXT NOT NULL DEFAULT '',
                platform TEXT NOT NULL DEFAULT '',
                app_version TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                revoked_at TEXT NULL,
                revoke_reason TEXT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_mobile_sessions_user
            ON mobile_sessions(user_id, revoked_at)
            """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mobile_refresh_tokens (
                token_hash TEXT PRIMARY KEY,
                family_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                consumed_at TEXT NULL,
                revoked_at TEXT NULL,
                replacement_hash TEXT NULL,
                FOREIGN KEY (session_id) REFERENCES mobile_sessions(session_id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_mobile_refresh_family
            ON mobile_refresh_tokens(family_id)
            """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mobile_message_requests (
                user_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                response_json TEXT NULL,
                error_code TEXT NULL,
                PRIMARY KEY (user_id, message_id)
            )
            """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_mobile_message_requests_created
            ON mobile_message_requests(created_at)
            """)
        conn.execute("COMMIT")
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        conn.close()


__all__ = [
    "connect",
    "default_users_db_path",
    "migrate_users_db",
]
