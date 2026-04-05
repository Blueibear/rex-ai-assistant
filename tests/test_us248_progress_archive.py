from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT / "docs" / "archive" / "progress"
ARCHIVED_PROGRESS_FILES = (
    "progress-ci-fix-pr216.txt",
    "progress.txt",
    "progress-full-repo-audit.txt",
    "progress-full-test-and-fix.txt",
    "progress-gui-autonomy-integrations.txt",
    "progress-master-next-cycle.txt",
    "progress-openclaw-http-integration.txt",
    "progress-openclaw-pivot - Copy.txt",
    "progress-openclaw-pivot-for-rex.txt",
    "progress-openclaw-pivot.txt",
    "progress-repo-quality.txt",
    "progress-voice-selector-and-fixes.txt",
)


def test_progress_logs_live_under_archive_directory() -> None:
    assert ARCHIVE_DIR.is_dir()

    for filename in ARCHIVED_PROGRESS_FILES:
        assert (ARCHIVE_DIR / filename).is_file()


def test_root_has_no_progress_text_files() -> None:
    assert not list(ROOT.glob("progress*.txt"))


def test_gitignore_blocks_new_progress_text_files_in_root() -> None:
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "progress*.txt" in text
