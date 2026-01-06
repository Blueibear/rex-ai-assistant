"""Repository-level invariants that guard against binary merge conflicts."""

from __future__ import annotations

import subprocess

_BINARY_SUFFIXES = (
    ".wav",
    ".mp3",
    ".onnx",
    ".pth",
    ".ckpt",
)


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
        if path.lower().endswith(_BINARY_SUFFIXES) or _is_binary(path):
            offenders.append(path)

    assert not offenders, (
        "Binary assets must not be tracked by Git. "
        "Remove these files or update .gitignore: "
        f"{', '.join(offenders)}"
    )
