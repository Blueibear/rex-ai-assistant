from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GITIGNORE = ROOT / ".gitignore"
ARCHIVED_PATCH = ROOT / "docs" / "archive" / "housekeeping" / "ci-fixes.patch"
ARCHIVE_NOTE = ROOT / "docs" / "archive" / "housekeeping" / "ci-fixes.patch.md"


def test_us251_patch_files_are_not_left_at_repo_root() -> None:
    root_patch_files = sorted(path.name for path in ROOT.glob("*.patch"))
    assert root_patch_files == []


def test_us251_patch_artifact_is_archived_with_note() -> None:
    assert ARCHIVED_PATCH.exists()
    assert ARCHIVE_NOTE.exists()


def test_gitignore_contains_patch_rule() -> None:
    text = GITIGNORE.read_text(encoding="utf-8")
    assert "*.patch" in text
