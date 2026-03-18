"""Repository-level invariants that guard against binary merge conflicts."""

from __future__ import annotations

import subprocess

from git_helpers import get_dirty_files

_BINARY_SUFFIXES = (
    ".wav",
    ".mp3",
    ".onnx",
    ".pth",
    ".ckpt",
)

# Small UI image assets (icons, logos) are intentionally tracked — they are
# not large ML artifacts and are needed for the app to function correctly.
_ALLOWED_BINARY_PREFIXES = ("gui/assets/",)


def _is_binary(path: str) -> bool:
    """Return ``True`` if the file contains non-textual bytes."""

    # GitHub flags conflicts for binary files even when the extension is not
    # on a well-known list, so fall back to a simple null-byte heuristic. The
    # files in this repository are small, making it safe to read them eagerly.
    with open(path, "rb") as handle:
        return b"\0" in handle.read()


def _list_tracked_files() -> list[str]:
    """Return all file paths tracked in the current Git commit."""

    completed = subprocess.run(
        ["git", "ls-tree", "-r", "HEAD", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def test_no_tracked_binary_artifacts():
    """Ensure that large binary assets stay out of version control."""

    offenders = []

    for path in _list_tracked_files():
        if any(path.startswith(prefix) for prefix in _ALLOWED_BINARY_PREFIXES):
            continue
        if path.lower().endswith(_BINARY_SUFFIXES) or _is_binary(path):
            offenders.append(path)

    assert not offenders, (
        "Binary assets must not be tracked by Git. "
        "Remove these files or update .gitignore: "
        f"{', '.join(offenders)}"
    )


def test_no_tracked_files_modified(tracked_modifications_baseline):
    """Ensure that no tracked file was modified during the test run.

    Tests must use tmp_path or temp copies for writable data.  Writing to
    tracked repo files (e.g. data/mock_calendar.json) breaks this invariant.
    """
    modified = [
        line
        for line in get_dirty_files(exclude_coverage=False)
        if line[0:2].strip().startswith("M") and line[3:] not in tracked_modifications_baseline
    ]
    assert not modified, (
        "Tests must not modify tracked files. Use tmp_path or temp copies "
        "for writable data. Dirty files: " + ", ".join(line.strip() for line in modified)
    )
