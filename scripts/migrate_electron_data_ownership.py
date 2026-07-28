"""Assign quarantined legacy Electron data to one explicitly chosen owner.

Dry-run is the default. Use ``--apply`` only after reviewing the counts. Each
changed store receives a one-time ``.pre-ownership-migration`` backup.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from rex.identity import validate_user_id


def _backup(path: Path) -> Path:
    backup = path.with_name(f"{path.name}.pre-ownership-migration")
    if backup.exists():
        raise FileExistsError(f"Migration backup already exists: {backup}")
    shutil.copy2(path, backup)
    return backup


def migrate_tasks(path: Path, user_id: str, *, apply: bool) -> int:
    if not path.exists():
        return 0
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    jobs = raw.get("jobs") if isinstance(raw, dict) else raw
    if not isinstance(jobs, list):
        raise ValueError(f"Unsupported scheduler data in {path}")

    unowned = [
        job
        for job in jobs
        if isinstance(job, dict)
        and not (isinstance(job.get("metadata"), dict) and job["metadata"].get("owner_user_id"))
    ]
    if apply and unowned:
        _backup(path)
        for job in unowned:
            metadata = job.setdefault("metadata", {})
            metadata["owner_user_id"] = user_id
            metadata["data_scope"] = "private"
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        temporary.replace(path)
    return len(unowned)


def migrate_history(path: Path, user_id: str, *, apply: bool) -> int:
    if not path.exists():
        return 0
    with sqlite3.connect(path) as conn:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(command_history)")}
        if "user_id" not in columns:
            if not apply:
                return int(conn.execute("SELECT COUNT(*) FROM command_history").fetchone()[0])
            _backup(path)
            conn.execute("ALTER TABLE command_history ADD COLUMN user_id TEXT")
        count = int(
            conn.execute(
                "SELECT COUNT(*) FROM command_history WHERE user_id IS NULL OR user_id = ''"
            ).fetchone()[0]
        )
        if apply and count:
            backup = path.with_name(f"{path.name}.pre-ownership-migration")
            if not backup.exists():
                _backup(path)
            conn.execute(
                "UPDATE command_history SET user_id = ? WHERE user_id IS NULL OR user_id = ''",
                (user_id,),
            )
            conn.commit()
        return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="Explicit owner for all quarantined rows")
    parser.add_argument("--apply", action="store_true", help="Write changes after creating backups")
    parser.add_argument("--scheduler-file", type=Path, default=Path("data/scheduler/jobs.json"))
    parser.add_argument("--history-db", type=Path, default=Path("data/command_history.db"))
    args = parser.parse_args(argv)

    user_id = validate_user_id(args.user)
    task_count = migrate_tasks(args.scheduler_file, user_id, apply=args.apply)
    history_count = migrate_history(args.history_db, user_id, apply=args.apply)
    mode = "APPLIED" if args.apply else "DRY RUN"
    print(f"{mode}: {task_count} task(s), {history_count} history row(s) -> {user_id}")
    if not args.apply and (task_count or history_count):
        print("Review the owner, then rerun with --apply. Unowned data remains quarantined.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
