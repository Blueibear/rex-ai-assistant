from __future__ import annotations

from pathlib import Path

import pytest

from scripts.migrate_runtime_data import (
    BACKUP_SUFFIX,
    MigrationItem,
    default_migration_items,
    main,
    migrate_items,
)


def test_dry_run_reports_plan_without_writing(tmp_path: Path) -> None:
    source = tmp_path / "legacy" / "history.db"
    target = tmp_path / "runtime" / "data" / "household" / "history.db"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"history")

    results = migrate_items([MigrationItem(source, target, "household")])

    assert [result.status for result in results] == ["planned"]
    assert not target.exists()
    assert not source.with_name(source.name + BACKUP_SUFFIX).exists()


def test_apply_backs_up_copies_and_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "legacy" / "scheduler"
    target = tmp_path / "runtime" / "data" / "household" / "scheduler"
    source.mkdir(parents=True)
    (source / "jobs.json").write_text('{"jobs": []}', encoding="utf-8")
    item = MigrationItem(source, target, "household")

    first = migrate_items([item], apply=True)
    second = migrate_items([item], apply=True)

    backup = source.with_name(source.name + BACKUP_SUFFIX)
    assert first[0].status == "migrated"
    assert first[0].backup == str(backup.resolve())
    assert (target / "jobs.json").read_text(encoding="utf-8") == '{"jobs": []}'
    assert (backup / "jobs.json").read_text(encoding="utf-8") == '{"jobs": []}'
    assert source.is_dir()
    assert second[0].status == "already_migrated"


def test_conflict_never_overwrites_target(tmp_path: Path) -> None:
    source = tmp_path / "legacy.json"
    target = tmp_path / "canonical.json"
    source.write_text("legacy", encoding="utf-8")
    target.write_text("canonical", encoding="utf-8")

    result = migrate_items(
        [MigrationItem(source, target, "household")],
        apply=True,
    )[0]

    assert result.status == "conflict"
    assert target.read_text(encoding="utf-8") == "canonical"
    assert not source.with_name(source.name + BACKUP_SUFFIX).exists()


def test_existing_backup_conflict_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "preferences.json"
    target = tmp_path / "private" / "preferences.json"
    backup = source.with_name(source.name + BACKUP_SUFFIX)
    source.write_text("current", encoding="utf-8")
    backup.write_text("different", encoding="utf-8")

    result = migrate_items(
        [MigrationItem(source, target, "private")],
        apply=True,
    )[0]

    assert result.status == "failed"
    assert "Backup conflict" in (result.detail or "")
    assert not target.exists()


def test_default_inventory_separates_private_and_household_data(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    home = tmp_path / "home"

    items = default_migration_items("james", root=root, home=home)
    pairs = {(item.source, item.target, item.scope) for item in items}

    assert (
        root / "data" / "users.db",
        root / "data" / "household" / "users.db",
        "household",
    ) in pairs
    assert (
        root / "data" / "memory" / "james",
        root / "data" / "users" / "james" / "memory",
        "private",
    ) in pairs
    assert (
        home / ".rex" / "preferences.json",
        root / "data" / "users" / "james" / "autonomy" / "preferences.json",
        "private",
    ) in pairs


def test_default_inventory_rejects_unsafe_user_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        default_migration_items("../other-user", root=tmp_path)


def test_cli_defaults_to_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    legacy = tmp_path / "data" / "users.db"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"users")

    exit_code = main(["--user", "james", "--runtime-root", str(tmp_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "DRY RUN" in output
    assert "[planned]" in output
    assert not (tmp_path / "data" / "household" / "users.db").exists()
