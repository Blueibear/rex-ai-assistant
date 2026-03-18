"""Shared git-status helpers used by repo integrity tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_dirty_files(exclude_coverage: bool = True) -> list[str]:
    """Return non-untracked git status lines for the repo working tree.

    Each line is in ``git status --porcelain`` format, e.g. ``" M path/to/file"``.
    Lines starting with ``??`` (untracked files) are excluded.
    Coverage artefacts are excluded from the git pathspec when *exclude_coverage*
    is ``True``.
    """
    args = ["git", "status", "--porcelain"]
    if exclude_coverage:
        args += ["--", ":!.coverage", ":!coverage.xml", ":!htmlcov/"]
    result = subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    return [line for line in result.stdout.splitlines() if line and not line.startswith("??")]
