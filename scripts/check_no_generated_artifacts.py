#!/usr/bin/env python3
"""Fail if generated build/test artifacts are committed (US-035).

Enumerates tracked files via ``git ls-files`` and exits non-zero when any
match a generated-artifact pattern (coverage output, build/dist trees,
Python bytecode caches). CI runs this as a blocking gate so accidental
``git add -A`` commits of local build output are rejected.

Usage:
    python scripts/check_no_generated_artifacts.py
"""

from __future__ import annotations

import re
import subprocess
import sys

# Each entry is (compiled pattern, human-readable description). Patterns are
# matched against git-tracked paths (always forward-slash separated).
# Deliberately tracked paths that would otherwise match a generated pattern.
# Every entry must state why it is committed on purpose.
ALLOWLIST: dict[str, str] = {
    # Served by rex-gui at /ui/ (rex/gui_app.py) so the developer-only
    # dashboard works from a plain `pip install .` without a Node build.
    "rex/ui/dist/index.html": "developer dashboard served by rex-gui",
}

GENERATED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(^|/)\.coverage($|\.)"), "coverage data file (.coverage)"),
    (re.compile(r"(^|/)coverage\.(xml|txt|html|json)$"), "coverage report"),
    (re.compile(r"(^|/)htmlcov/"), "HTML coverage report directory"),
    (re.compile(r"(^|/)dist/"), "build distribution directory (dist/)"),
    (re.compile(r"(^|/)build/"), "build output directory (build/)"),
    (re.compile(r"(^|/)__pycache__/"), "Python bytecode cache (__pycache__/)"),
    (re.compile(r"\.py[co]$"), "compiled Python bytecode (*.pyc / *.pyo)"),
    (re.compile(r"(^|/)[^/]+\.egg-info/"), "setuptools metadata (*.egg-info/)"),
]


def classify_paths(paths: list[str]) -> list[tuple[str, str]]:
    """Return (path, description) for every generated artifact in *paths*."""
    offenders: list[tuple[str, str]] = []
    for path in paths:
        if not path or path in ALLOWLIST:
            continue
        for pattern, description in GENERATED_PATTERNS:
            if pattern.search(path):
                offenders.append((path, description))
                break
    return offenders


def find_tracked_generated_artifacts() -> list[tuple[str, str]]:
    """Return (path, description) for every tracked generated artifact."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        check=True,
        text=True,
    )
    return classify_paths(result.stdout.split("\0"))


def main() -> int:
    offenders = find_tracked_generated_artifacts()
    if offenders:
        print("ERROR: generated artifacts are committed to the repository:")
        for path, description in offenders:
            print(f"  {path}  ({description})")
        print(
            "Remove them with 'git rm --cached <path>' and ensure .gitignore " "covers the pattern."
        )
        return 1
    print("OK: no generated artifacts are tracked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
