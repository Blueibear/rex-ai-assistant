from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.migrate_electron_data_ownership import migrate_history, migrate_tasks


def test_ownership_migration_is_dry_run_by_default(tmp_path: Path) -> None:
    tasks = tmp_path / "jobs.json"
    tasks.write_text(json.dumps([{"job_id": "legacy", "metadata": {}}]), encoding="utf-8")
    before = tasks.read_bytes()

    assert migrate_tasks(tasks, "james", apply=False) == 1
    assert tasks.read_bytes() == before
    assert not tasks.with_name(f"{tasks.name}.pre-ownership-migration").exists()


def test_ownership_migration_backs_up_and_assigns_explicit_owner(tmp_path: Path) -> None:
    tasks = tmp_path / "jobs.json"
    tasks.write_text(json.dumps([{"job_id": "legacy", "metadata": {}}]), encoding="utf-8")
    history = tmp_path / "history.db"
    with sqlite3.connect(history) as conn:
        conn.execute(
            "CREATE TABLE command_history (id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, "
            "command TEXT NOT NULL, result TEXT, success INTEGER NOT NULL DEFAULT 1)"
        )
        conn.execute(
            "INSERT INTO command_history (timestamp, command, result, success) "
            "VALUES ('2026-01-01', 'legacy', '', 1)"
        )

    assert migrate_tasks(tasks, "cole", apply=True) == 1
    assert migrate_history(history, "cole", apply=True) == 1

    payload = json.loads(tasks.read_text(encoding="utf-8"))
    assert payload[0]["metadata"]["owner_user_id"] == "cole"
    with sqlite3.connect(history) as conn:
        assert conn.execute("SELECT user_id FROM command_history").fetchone()[0] == "cole"
    assert tasks.with_name(f"{tasks.name}.pre-ownership-migration").exists()
    assert history.with_name(f"{history.name}.pre-ownership-migration").exists()
