"""Database migration state validation.

Tracks which schema migrations have been applied and validates that all
expected migrations are present before the application accepts traffic.

Environment variables
---------------------
SKIP_MIGRATION_CHECK    Set to "1", "true", or "yes" to bypass the check
                        (emergency use only).

Usage
-----
Call :func:`validate_migration_state` early in application startup, before
any request handler is registered::

    from rex.migrations import validate_migration_state
    validate_migration_state()

If unapplied migrations exist the function logs each pending migration name
at WARNING level and exits the process with code 1.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default database path (matches DashboardStore default)
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = Path("data/dashboard_notifications.db")

# ---------------------------------------------------------------------------
# Migration registry
# Each entry is a (name, sql) tuple.  Names must be unique and are stored in
# the schema_migrations table when applied.
# ---------------------------------------------------------------------------

MIGRATIONS: list[tuple[str, str]] = [
    (
        "001_create_notifications",
        """
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
        """,
    ),
    (
        "002_idx_notifications_timestamp",
        """
        CREATE INDEX IF NOT EXISTS idx_notifications_timestamp
        ON notifications (timestamp DESC)
        """,
    ),
    (
        "003_idx_notifications_user_id",
        """
        CREATE INDEX IF NOT EXISTS idx_notifications_user_id
        ON notifications (user_id)
        """,
    ),
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SCHEMA_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    name TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
)
"""

_EXPECTED_NAMES: tuple[str, ...] = tuple(name for name, _ in MIGRATIONS)


def _open_db(db_path: Path) -> sqlite3.Connection:
    """Open a direct (non-pooled) connection for migration checks."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_pending_migrations(db_path: Path | None = None) -> list[str]:
    """Return names of migrations that have not yet been applied.

    Creates the ``schema_migrations`` tracking table if it does not exist so
    that bootstrapping a fresh database works without a separate step.

    Args:
        db_path: Path to the SQLite database file.  Defaults to the
            dashboard store default path.

    Returns:
        A list of migration names (in order) that are not recorded in
        ``schema_migrations``.
    """
    resolved = db_path or _DEFAULT_DB_PATH
    conn = _open_db(resolved)
    try:
        conn.execute(_SCHEMA_TABLE_SQL)
        conn.commit()
        rows = conn.execute("SELECT name FROM schema_migrations").fetchall()
        applied: set[str] = {row["name"] for row in rows}
    finally:
        conn.close()

    return [name for name in _EXPECTED_NAMES if name not in applied]


def mark_migration_applied(name: str, db_path: Path | None = None) -> None:
    """Record a migration as applied in the tracking table.

    Args:
        name: Migration name (must be in :data:`MIGRATIONS`).
        db_path: Path to the SQLite database file.

    Raises:
        ValueError: If *name* is not a known migration.
    """
    if name not in _EXPECTED_NAMES:
        raise ValueError(f"Unknown migration: {name!r}")
    resolved = db_path or _DEFAULT_DB_PATH
    conn = _open_db(resolved)
    try:
        conn.execute(_SCHEMA_TABLE_SQL)
        applied_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (name, applied_at) VALUES (?, ?)",
            (name, applied_at),
        )
        conn.commit()
    finally:
        conn.close()


def validate_migration_state(db_path: Path | None = None) -> None:
    """Validate that all migrations have been applied.

    Should be called on application startup, before any request handler is
    registered.  Exits with code 1 if unapplied migrations are found (unless
    ``SKIP_MIGRATION_CHECK`` is set).

    Args:
        db_path: Path to the SQLite database file.  Defaults to the
            dashboard store default path.
    """
    skip = os.environ.get("SKIP_MIGRATION_CHECK", "").lower() in {"1", "true", "yes"}
    if skip:
        logger.warning("Migration check skipped (SKIP_MIGRATION_CHECK is set)")
        return

    pending = get_pending_migrations(db_path)
    if not pending:
        logger.debug("Migration state OK — all %d migrations applied", len(_EXPECTED_NAMES))
        return

    logger.warning(
        "Unapplied database migrations detected (%d pending). "
        "Run migrations before starting the application.",
        len(pending),
    )
    for name in pending:
        logger.warning("  Pending migration: %s", name)

    sys.exit(1)


__all__ = [
    "MIGRATIONS",
    "get_pending_migrations",
    "mark_migration_applied",
    "validate_migration_state",
]
