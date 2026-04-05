from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GITIGNORE = ROOT / ".gitignore"
TRACKED_BACKUP_ENV_FILES = (
    ".env.backup-legacy",
    ".env.example.backup_before_refactor",
)
IGNORED_PATHS = (
    ".env.backup-legacy",
    ".env.backup.20260118_224707",
    ".env.example.backup.future",
    "backups",
    "backups/.env.backup.20260118_224707",
)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_gitignore_contains_us250_env_backup_rules() -> None:
    text = GITIGNORE.read_text(encoding="utf-8")

    for pattern in (".env.backup*", "*.env.backup*", ".env.example.backup*", "backups/"):
        assert pattern in text


def test_us250_backup_env_paths_are_ignored() -> None:
    for path in IGNORED_PATHS:
        result = _git("check-ignore", "-v", path)
        assert ".gitignore" in result.stdout, f"{path} should be ignored by .gitignore"


def test_us250_backup_env_files_are_not_tracked() -> None:
    result = _git("ls-files", "--", *TRACKED_BACKUP_ENV_FILES)
    tracked = [line for line in result.stdout.splitlines() if line]
    assert tracked == [], f"Expected no tracked backup env files, found: {tracked}"


def test_us250_only_env_example_is_tracked_at_repo_root() -> None:
    result = _git("ls-files", "--", ".env*")
    tracked = [line for line in result.stdout.splitlines() if line]
    assert tracked == [".env.example"], f"Expected only .env.example to be tracked, found: {tracked}"
