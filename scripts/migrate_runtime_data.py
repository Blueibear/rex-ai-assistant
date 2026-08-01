"""Dry-run-first migration into AskRex's canonical runtime data layout.

The migration never deletes a legacy source. Applying a migration creates an
adjacent backup, copies into an absent target atomically, and refuses to
replace conflicting data. Re-running the migration is safe and reports items
that already match their destination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from rex.identity import validate_user_id
from rex.runtime_paths import runtime_root

BACKUP_SUFFIX = ".pre-runtime-root-migration.bak"


@dataclass(frozen=True)
class MigrationItem:
    """One legacy source and its canonical destination."""

    source: Path
    target: Path
    scope: str


@dataclass(frozen=True)
class MigrationResult:
    """Outcome for one migration item."""

    source: str
    target: str
    scope: str
    status: str
    backup: str | None = None
    detail: str | None = None


def default_migration_items(
    user_id: str,
    *,
    root: Path | None = None,
    home: Path | None = None,
) -> list[MigrationItem]:
    """Build the known legacy-to-canonical migration inventory."""
    owner = validate_user_id(user_id)
    base = (root or runtime_root()).resolve(strict=False)
    home_root = (home or Path.home()).resolve(strict=False)
    legacy_data = base / "data"
    household = legacy_data / "household"
    private = legacy_data / "users" / owner

    household_files = (
        "users.db",
        "history.db",
        "command_history.db",
        "dashboard_notifications.db",
        "llm_usage.json",
        "shopping_list.json",
    )
    household_dirs = (
        "approvals",
        "automations",
        "browser_sessions",
        "cues",
        "knowledge_base",
        "notifications",
        "reminders",
        "scheduler",
        "workflows",
    )

    items = [
        MigrationItem(legacy_data / name, household / name, "household") for name in household_files
    ]
    items.extend(
        MigrationItem(legacy_data / name, household / name, "household") for name in household_dirs
    )
    items.extend(
        (
            MigrationItem(
                legacy_data / "memory" / owner,
                private / "memory",
                "private",
            ),
            MigrationItem(
                home_root / ".rex" / "preferences.json",
                private / "autonomy" / "preferences.json",
                "private",
            ),
            MigrationItem(
                home_root / ".rex" / "execution_history.db",
                private / "autonomy" / "execution_history.db",
                "private",
            ),
            MigrationItem(
                home_root / ".rex" / "notifications.db",
                household / "notifications.db",
                "household",
            ),
            MigrationItem(
                home_root / ".rex" / "contacts.json",
                household / "contacts.json",
                "household",
            ),
        )
    )
    return items


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(b"file\0")
        digest.update(path.read_bytes())
        return digest.hexdigest()
    if path.is_dir():
        digest.update(b"directory\0")
        for child in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
            relative = child.relative_to(path).as_posix().encode("utf-8")
            digest.update(relative)
            digest.update(b"\0")
            if child.is_symlink():
                digest.update(b"symlink\0")
                digest.update(os.readlink(child).encode("utf-8"))
            elif child.is_file():
                digest.update(b"file\0")
                digest.update(child.read_bytes())
            elif child.is_dir():
                digest.update(b"directory\0")
        return digest.hexdigest()
    raise ValueError(f"Unsupported migration source: {path}")


def _same_content(source: Path, target: Path) -> bool:
    if source.is_file() != target.is_file() or source.is_dir() != target.is_dir():
        return False
    return _digest(source) == _digest(target)


def _backup_path(source: Path) -> Path:
    return source.with_name(source.name + BACKUP_SUFFIX)


def _copy_absent(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.migrating.{os.getpid()}")
    if temporary.exists():
        if temporary.is_dir():
            shutil.rmtree(temporary)
        else:
            temporary.unlink()
    try:
        if source.is_dir():
            shutil.copytree(source, temporary, symlinks=True)
            temporary.rename(target)
        else:
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
    finally:
        if temporary.exists():
            if temporary.is_dir():
                shutil.rmtree(temporary)
            else:
                temporary.unlink()


def _ensure_backup(source: Path) -> Path:
    backup = _backup_path(source)
    if backup.exists():
        if not _same_content(source, backup):
            raise FileExistsError(f"Backup conflict: {backup}")
        return backup
    if source.is_dir():
        shutil.copytree(source, backup, symlinks=True)
    else:
        shutil.copy2(source, backup)
    return backup


def migrate_items(
    items: Iterable[MigrationItem],
    *,
    apply: bool = False,
) -> list[MigrationResult]:
    """Plan or apply migration items without overwriting any destination."""
    results: list[MigrationResult] = []
    for item in items:
        source = item.source.resolve(strict=False)
        target = item.target.resolve(strict=False)
        common = {
            "source": str(source),
            "target": str(target),
            "scope": item.scope,
        }
        if source == target:
            results.append(MigrationResult(**common, status="already_canonical"))
            continue
        if not source.exists():
            results.append(MigrationResult(**common, status="missing"))
            continue
        if source.is_symlink():
            results.append(
                MigrationResult(
                    **common,
                    status="unsupported",
                    detail="Top-level symbolic links are not migrated",
                )
            )
            continue
        if target.exists():
            status = "already_migrated" if _same_content(source, target) else "conflict"
            results.append(MigrationResult(**common, status=status))
            continue
        if not apply:
            results.append(MigrationResult(**common, status="planned"))
            continue
        try:
            backup = _ensure_backup(source)
            _copy_absent(source, target)
        except (OSError, ValueError) as exc:
            results.append(MigrationResult(**common, status="failed", detail=str(exc)))
            continue
        results.append(
            MigrationResult(
                **common,
                status="migrated",
                backup=str(backup.resolve(strict=False)),
            )
        )
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="Owner for legacy private data")
    parser.add_argument("--runtime-root", type=Path, help="Override the AskRex runtime root")
    parser.add_argument("--home", type=Path, help="Override the legacy home directory")
    parser.add_argument("--apply", action="store_true", help="Create backups and copy data")
    parser.add_argument("--json", action="store_true", help="Print machine-readable results")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    items = default_migration_items(args.user, root=args.runtime_root, home=args.home)
    results = migrate_items(items, apply=args.apply)
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        mode = "APPLY" if args.apply else "DRY RUN"
        print(f"AskRex runtime data migration — {mode}")
        for result in results:
            detail = f" ({result.detail})" if result.detail else ""
            print(f"[{result.status}] {result.source} -> {result.target}{detail}")
    return (
        1
        if any(result.status in {"conflict", "failed", "unsupported"} for result in results)
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
