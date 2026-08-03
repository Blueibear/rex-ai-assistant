"""Migration tests: fresh, legacy, and repeated runs against users.db.

Matrix rows: USR-001, USR-002, USR-003, USR-004.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import bcrypt
import pytest

from tests.mobile_api.conftest import login

pytestmark = pytest.mark.usefixtures("mobile_env")


def _table_names(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    finally:
        conn.close()
    return {row[0] for row in rows}


def _table_columns(db_path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    return {row[1] for row in rows}


def _user_columns(db_path: Path) -> set[str]:
    return _table_columns(db_path, "users")


def _make_legacy_db(db_path: Path, username: str, password: str) -> str:
    """Create a pre-mobile users.db (no disabled_at, no mobile tables)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE users (
                id       TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created  TEXT NOT NULL
            )
            """)
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        user_id = "11111111-2222-3333-4444-555555555555"
        conn.execute(
            "INSERT INTO users (id, username, password, created) VALUES (?, ?, ?, ?)",
            (user_id, username, password_hash, datetime.now(UTC).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    return user_id


class TestFreshMigration:
    def test_creates_all_tables_and_columns(self, mobile_env: Path) -> None:
        from rex.mobile_api.db import migrate_users_db

        db_path = mobile_env / "users.db"
        migrate_users_db(db_path)

        tables = _table_names(db_path)
        assert {
            "users",
            "user_permissions",
            "mobile_sessions",
            "mobile_refresh_tokens",
            "mobile_paired_devices",
            "mobile_device_grants",
            "mobile_device_session_challenges",
            "mobile_strong_auth_challenges",
            "mobile_strong_auth_audit",
        } <= tables
        assert "disabled_at" in _user_columns(db_path)
        assert "last_strong_auth_at" in _table_columns(db_path, "mobile_device_grants")
        assert {
            "action_hash",
            "risk_level",
            "approval_id",
            "consumed_at",
        } <= _table_columns(db_path, "mobile_strong_auth_challenges")
        audit_columns = _table_columns(db_path, "mobile_strong_auth_audit")
        assert {"action_hash", "event_type", "reason"} <= audit_columns
        assert "payload" not in audit_columns
        assert "signature" not in audit_columns


class TestLegacyMigration:
    def test_preserves_existing_users_and_hashes(self, mobile_env: Path) -> None:
        from rex.mobile_api.db import migrate_users_db

        db_path = mobile_env / "users.db"
        user_id = _make_legacy_db(db_path, "james", "legacy-pass")

        before = sqlite3.connect(str(db_path))
        original_hash = before.execute(
            "SELECT password FROM users WHERE id = ?", (user_id,)
        ).fetchone()[0]
        before.close()

        migrate_users_db(db_path)

        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT username, password, disabled_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert row[0] == "james"
        assert row[1] == original_hash
        assert row[2] is None  # existing users remain active

    def test_existing_user_logs_in_after_migration(
        self, mobile_env: Path, clock, mobile_config
    ) -> None:
        from rex.mobile_api.app import create_mobile_app
        from rex.mobile_api.services import MobileApiServices

        db_path = mobile_env / "users.db"
        _make_legacy_db(db_path, "james", "legacy-pass")

        services = MobileApiServices.build(mobile_config, db_path=db_path, clock=clock)
        app = create_mobile_app(services=services)
        app.config["TESTING"] = True
        with app.test_client() as client:
            response = login(client, "james", "legacy-pass")
        assert response.status_code == 200
        assert "access_token" in response.get_json()


class TestRepeatedMigration:
    def test_migration_is_idempotent(self, mobile_env: Path) -> None:
        from rex.mobile_api.db import migrate_users_db

        db_path = mobile_env / "users.db"
        migrate_users_db(db_path)
        first = _table_names(db_path)
        migrate_users_db(db_path)
        migrate_users_db(db_path)
        assert _table_names(db_path) == first
        assert "disabled_at" in _user_columns(db_path)
