from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
SCRIPTS_README = SCRIPTS_DIR / "README.md"

# The utility scripts that US-252 required to be removed from root.
# Each was either moved to scripts/ or deleted (wake_acknowledgment.py).
MOVED_SCRIPTS = (
    "check_gpu_status.py",
    "check_imports.py",
    "check_patch_status.py",
    "check_tts_imports.py",
    "find_gpt2_model.py",
    "generate_wake_sound.py",
    "list_voices.py",
    "manual_search_demo.py",
    "manual_whisper_demo.py",
    "play_test.py",
    "record_wakeword.py",
    "test_imports.py",
    "test_transformers_patch.py",
)

# Scripts that were added directly to scripts/ (not moved from root)
NEW_IN_SCRIPTS = (
    "list_audio.py",
    "test_mic_open.py",
)

# Script that was deleted entirely
DELETED_SCRIPTS = ("wake_acknowledgment.py",)


def test_us252_listed_scripts_not_at_root() -> None:
    """All utility scripts from the US-252 explicit list must be gone from root."""
    present_at_root = [
        name
        for name in (*MOVED_SCRIPTS, *DELETED_SCRIPTS)
        if (ROOT / name).exists()
    ]
    assert present_at_root == [], (
        f"These scripts should no longer be at repo root: {present_at_root}"
    )


def test_us252_moved_scripts_exist_in_scripts_dir() -> None:
    """Scripts that were moved must exist under scripts/."""
    missing = [
        name for name in MOVED_SCRIPTS if not (SCRIPTS_DIR / name).exists()
    ]
    assert missing == [], (
        f"These scripts should exist under scripts/ but do not: {missing}"
    )


def test_us252_new_scripts_exist_in_scripts_dir() -> None:
    """Scripts newly added for scripts/ must be present."""
    missing = [
        name for name in NEW_IN_SCRIPTS if not (SCRIPTS_DIR / name).exists()
    ]
    assert missing == [], (
        f"These new scripts should exist under scripts/ but do not: {missing}"
    )


def test_us252_scripts_readme_documents_moved_scripts() -> None:
    """scripts/README.md must mention every moved script."""
    assert SCRIPTS_README.exists(), "scripts/README.md must exist"
    text = SCRIPTS_README.read_text(encoding="utf-8")
    missing_docs = [name for name in MOVED_SCRIPTS if name not in text]
    assert missing_docs == [], (
        f"scripts/README.md is missing documentation for: {missing_docs}"
    )
