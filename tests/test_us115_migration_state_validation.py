"""Tests for US-115: Migration state validation on startup.

Acceptance criteria:
- on startup, the migration state is queried and compared against the
  expected schema version
- if unapplied migrations exist, the application logs the pending migration
  names and exits with code 1
- migration check runs before any request handler is registered
- check can be disabled via a SKIP_MIGRATION_CHECK environment variable for
  emergency use
- Typecheck passes
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rex.migrations import (
    MIGRATIONS,
    get_pending_migrations,
    mark_migration_applied,
    validate_migration_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _db_with_all_applied(tmp_path: Path) -> Path:
    """Create a DB where all known migrations are recorded as applied."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            name TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    for name, _ in MIGRATIONS:
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (name, applied_at) VALUES (?, ?)",
            (name, "2024-01-01T00:00:00+00:00"),
        )
    conn.commit()
    conn.close()
    return db_path


def _db_with_none_applied(tmp_path: Path) -> Path:
    """Create a DB where schema_migrations exists but no migrations applied."""
    db_path = tmp_path / "test_empty.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            name TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()
    return db_path


def _fresh_db(tmp_path: Path) -> Path:
    """Return a path to a non-existent DB (no schema_migrations table)."""
    return tmp_path / "fresh.db"


# ---------------------------------------------------------------------------
# get_pending_migrations
# ---------------------------------------------------------------------------


class TestGetPendingMigrations:
    def test_all_applied_returns_empty(self, tmp_path: Path) -> None:
        db_path = _db_with_all_applied(tmp_path)
        assert get_pending_migrations(db_path) == []

    def test_none_applied_returns_all(self, tmp_path: Path) -> None:
        db_path = _db_with_none_applied(tmp_path)
        pending = get_pending_migrations(db_path)
        expected_names = [name for name, _ in MIGRATIONS]
        assert pending == expected_names

    def test_partial_returns_missing(self, tmp_path: Path) -> None:
        db_path = tmp_path / "partial.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            """
        )
        # Apply only the first migration
        first_name = MIGRATIONS[0][0]
        conn.execute(
            "INSERT INTO schema_migrations (name, applied_at) VALUES (?, ?)",
            (first_name, "2024-01-01T00:00:00+00:00"),
        )
        conn.commit()
        conn.close()

        pending = get_pending_migrations(db_path)
        assert first_name not in pending
        for name, _ in MIGRATIONS[1:]:
            assert name in pending

    def test_fresh_db_creates_schema_migrations_table(self, tmp_path: Path) -> None:
        """Bootstrap: schema_migrations should be created if it does not exist."""
        db_path = _fresh_db(tmp_path)
        assert not db_path.exists()

        pending = get_pending_migrations(db_path)

        # DB should now exist with schema_migrations table
        assert db_path.exists()
        conn = sqlite3.connect(str(db_path))
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()
        assert "schema_migrations" in tables
        # All migrations are pending since none have been applied
        assert len(pending) == len(MIGRATIONS)

    def test_returns_pending_in_order(self, tmp_path: Path) -> None:
        """Pending names are returned in migration definition order."""
        db_path = _db_with_none_applied(tmp_path)
        pending = get_pending_migrations(db_path)
        expected = [name for name, _ in MIGRATIONS]
        assert pending == expected


# ---------------------------------------------------------------------------
# mark_migration_applied
# ---------------------------------------------------------------------------


class TestMarkMigrationApplied:
    def test_marks_migration_applied(self, tmp_path: Path) -> None:
        db_path = _fresh_db(tmp_path)
        first_name = MIGRATIONS[0][0]
        mark_migration_applied(first_name, db_path)
        pending = get_pending_migrations(db_path)
        assert first_name not in pending

    def test_unknown_name_raises(self, tmp_path: Path) -> None:
        db_path = _fresh_db(tmp_path)
        with pytest.raises(ValueError, match="Unknown migration"):
            mark_migration_applied("999_nonexistent", db_path)

    def test_idempotent_double_apply(self, tmp_path: Path) -> None:
        db_path = _fresh_db(tmp_path)
        first_name = MIGRATIONS[0][0]
        mark_migration_applied(first_name, db_path)
        # Should not raise on a second call
        mark_migration_applied(first_name, db_path)
        pending = get_pending_migrations(db_path)
        assert first_name not in pending


# ---------------------------------------------------------------------------
# validate_migration_state
# ---------------------------------------------------------------------------


class TestValidateMigrationState:
    def test_no_pending_does_not_exit(self, tmp_path: Path) -> None:
        db_path = _db_with_all_applied(tmp_path)
        # Should not raise SystemExit
        validate_migration_state(db_path)

    def test_pending_exits_with_code_1(self, tmp_path: Path) -> None:
        db_path = _db_with_none_applied(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            validate_migration_state(db_path)
        assert exc_info.value.code == 1

    def test_pending_logs_migration_names(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        db_path = _db_with_none_applied(tmp_path)
        with pytest.raises(SystemExit):
            with caplog.at_level("WARNING"):
                validate_migration_state(db_path)
        # Each pending migration name should appear in the log output
        for name, _ in MIGRATIONS:
            assert name in caplog.text

    def test_skip_env_var_bypasses_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SKIP_MIGRATION_CHECK=1 suppresses exit even with pending migrations."""
        monkeypatch.setenv("SKIP_MIGRATION_CHECK", "1")
        db_path = _db_with_none_applied(tmp_path)
        # Should not raise SystemExit
        validate_migration_state(db_path)

    def test_skip_env_var_true_bypasses_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SKIP_MIGRATION_CHECK", "true")
        db_path = _db_with_none_applied(tmp_path)
        validate_migration_state(db_path)

    def test_skip_env_var_yes_bypasses_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SKIP_MIGRATION_CHECK", "yes")
        db_path = _db_with_none_applied(tmp_path)
        validate_migration_state(db_path)

    def test_skip_env_var_not_set_does_not_bypass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SKIP_MIGRATION_CHECK", raising=False)
        db_path = _db_with_none_applied(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            validate_migration_state(db_path)
        assert exc_info.value.code == 1

    def test_partial_pending_exits(self, tmp_path: Path) -> None:
        """Even one unapplied migration triggers exit."""
        db_path = tmp_path / "partial.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            """
        )
        # Apply all but the last migration
        for name, _ in MIGRATIONS[:-1]:
            conn.execute(
                "INSERT INTO schema_migrations (name, applied_at) VALUES (?, ?)",
                (name, "2024-01-01T00:00:00+00:00"),
            )
        conn.commit()
        conn.close()

        with pytest.raises(SystemExit) as exc_info:
            validate_migration_state(db_path)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Migration registry completeness
# ---------------------------------------------------------------------------


class TestMigrationRegistry:
    def test_migrations_list_is_nonempty(self) -> None:
        assert len(MIGRATIONS) > 0

    def test_migration_names_are_unique(self) -> None:
        names = [name for name, _ in MIGRATIONS]
        assert len(names) == len(set(names)), "Duplicate migration names detected"

    def test_migration_entries_are_tuples_of_two(self) -> None:
        for entry in MIGRATIONS:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            name, sql = entry
            assert isinstance(name, str) and name
            assert isinstance(sql, str) and sql.strip()


# ---------------------------------------------------------------------------
# flask_proxy integration: check runs before blueprints
# ---------------------------------------------------------------------------


class TestFlaskProxyIntegration:
    def test_validate_called_before_blueprints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """validate_migration_state is called before blueprint registration in flask_proxy."""
        import importlib
        import sys

        call_order: list[str] = []

        # Patch validate_migration_state to record call order
        import rex.migrations as mig_mod

        original_validate = mig_mod.validate_migration_state

        def fake_validate(db_path: Path | None = None) -> None:
            call_order.append("validate")

        monkeypatch.setattr(mig_mod, "validate_migration_state", fake_validate)

        # Patch Flask.register_blueprint to record call order
        from flask import Flask

        original_register = Flask.register_blueprint

        def fake_register(self: Flask, blueprint: object, **kwargs: object) -> None:
            call_order.append("register_blueprint")
            return original_register(self, blueprint, **kwargs)  # type: ignore[return-value]

        monkeypatch.setattr(Flask, "register_blueprint", fake_register)

        # Re-import flask_proxy so the patched functions are used
        monkeypatch.setenv("REX_TESTING", "1")
        monkeypatch.setenv("SKIP_MIGRATION_CHECK", "1")
        if "flask_proxy" in sys.modules:
            del sys.modules["flask_proxy"]

        importlib.import_module("flask_proxy")

        # validate must appear before the first register_blueprint call
        assert "validate" in call_order
        first_validate = call_order.index("validate")
        blueprint_calls = [i for i, v in enumerate(call_order) if v == "register_blueprint"]
        assert blueprint_calls, "No register_blueprint calls found"
        assert first_validate < blueprint_calls[0], (
            f"validate called at index {first_validate} but first "
            f"register_blueprint at {blueprint_calls[0]}"
        )

        monkeypatch.setattr(mig_mod, "validate_migration_state", original_validate)
