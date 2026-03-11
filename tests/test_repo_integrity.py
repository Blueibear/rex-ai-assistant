"""Guard test: ensure the test suite does not modify tracked repository files.

Tests must use ``tmp_path`` or ``tempfile`` for any file writes.  Writing to
tracked paths (e.g. ``data/mock_calendar.json``) dirties the working tree and
can cause spurious git diffs for other developers.

This test runs ``git status --porcelain`` and fails if any tracked file was
modified.  Coverage artefacts (``.coverage``, ``coverage.xml``, ``htmlcov/``)
are excluded because they are generated intentionally and are already in
``.gitignore``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestRepoIntegrity:
    """Working tree must remain clean after tests."""

    def test_no_tracked_files_modified(self, tracked_modifications_baseline):
        """No tracked file should be modified by the test suite."""
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                ":!.coverage",
                ":!coverage.xml",
                ":!htmlcov/",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        # Filter out untracked files (lines starting with '??')
        dirty = [
            line
            for line in result.stdout.splitlines()
            if line and not line.startswith("??") and line[3:] not in tracked_modifications_baseline
        ]
        assert dirty == [], (
            "Test suite modified tracked repo files.  Tests must write to "
            "tmp_path or tempfile, not to tracked paths.\n"
            "Dirty files:\n" + "\n".join(f"  {d}" for d in dirty)
        )
