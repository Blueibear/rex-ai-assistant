"""Guard test: ensure the test suite does not modify tracked repository files.

Tests must use ``tmp_path`` or ``tempfile`` for any file writes.  Writing to
tracked paths (e.g. ``data/mock_calendar.json``) dirties the working tree and
can cause spurious git diffs for other developers.

Baseline approach: the ``tracked_modifications_baseline`` session fixture in
``conftest.py`` captures a snapshot of all already-dirty tracked files via
``git status --porcelain`` *before* any test runs.  Each integrity test then
filters the current dirty set against that baseline so pre-existing modifications
(e.g. ``requirements-gpu-cu124.txt`` tweaked locally) do not cause false failures.
Only files that became dirty *during* the test run are flagged.

Coverage artefacts (``.coverage``, ``coverage.xml``, ``htmlcov/``) are
excluded because they are generated intentionally and are already in
``.gitignore``.
"""

from __future__ import annotations

from git_helpers import get_dirty_files


class TestRepoIntegrity:
    """Working tree must remain clean after tests."""

    def test_no_tracked_files_modified(self, tracked_modifications_baseline):
        """No tracked file should be modified by the test suite."""
        dirty = [
            line for line in get_dirty_files() if line[3:] not in tracked_modifications_baseline
        ]
        assert dirty == [], (
            "Test suite modified tracked repo files.  Tests must write to "
            "tmp_path or tempfile, not to tracked paths.\n"
            "Dirty files:\n" + "\n".join(f"  {d}" for d in dirty)
        )
