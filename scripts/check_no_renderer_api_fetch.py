#!/usr/bin/env python3
"""Guard against raw renderer fetch('/api/...') calls in gui/src/.

Scans gui/src/**/*.{ts,tsx,js,jsx} for patterns that match
  fetch('/api...   fetch("/api...   fetch(`/api...

Any match whose "relative/path:lineno" is NOT listed in
gui/src/ALLOWED_API_FETCHES.txt causes the script to exit 1.

Usage:
    python scripts/check_no_renderer_api_fetch.py [--repo-root PATH]

Exit codes:
    0  no unapproved raw /api/ fetches found
    1  one or more unapproved raw /api/ fetches found
"""

import argparse
import re
import sys
from pathlib import Path

_PATTERN = re.compile(r"""fetch\(['"` ]/api""".replace(" ", ""))
_GUI_SRC_GLOBS = ("*.ts", "*.tsx", "*.js", "*.jsx")
_ALLOWLIST_RELPATH = Path("gui/src/ALLOWED_API_FETCHES.txt")


def load_allowlist(repo_root: Path) -> set[str]:
    """Return set of 'rel/path:lineno' keys from the allowlist file."""
    path = repo_root / _ALLOWLIST_RELPATH
    if not path.exists():
        return set()
    entries: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#")[0].strip()
        if stripped:
            entries.add(stripped)
    return entries


def find_violations(repo_root: Path) -> list[tuple[str, int, str]]:
    """Return (rel_path, lineno, line_content) for each unapproved match."""
    allowlist = load_allowlist(repo_root)
    gui_src = repo_root / "gui" / "src"
    violations: list[tuple[str, int, str]] = []

    if not gui_src.is_dir():
        return violations

    for glob in _GUI_SRC_GLOBS:
        for fpath in sorted(gui_src.rglob(glob)):
            rel = fpath.relative_to(repo_root).as_posix()
            try:
                lines = fpath.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for i, line in enumerate(lines, start=1):
                if _PATTERN.search(line):
                    key = f"{rel}:{i}"
                    if key not in allowlist:
                        violations.append((rel, i, line.strip()))

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check for unapproved raw /api/ fetches in gui/src/."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root directory (default: current directory)",
    )
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()

    violations = find_violations(repo_root)
    if violations:
        print(f"ERROR: {len(violations)} unapproved raw /api/ fetch(s) in gui/src/:")
        for rel, lineno, content in violations:
            print(f"  {rel}:{lineno}  {content}")
        print()
        print(
            "To allow an existing call site, add a line to "
            "gui/src/ALLOWED_API_FETCHES.txt:\n"
            "  rel/path/to/file.tsx:lineno  # justification\n"
            "To fix permanently, migrate the call to typed IPC."
        )
        return 1

    print("OK: no unapproved raw /api/ fetches in gui/src/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
