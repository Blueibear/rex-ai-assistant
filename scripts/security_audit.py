#!/usr/bin/env python3
"""Security audit script for Rex AI Assistant repository.

Scans source files for:
- Merge conflict markers
- Placeholder/incomplete code markers
- Exposed secrets and API keys

Excludes: .git, .venv, venv, source, node_modules, dist, build, __pycache__, backups, logs

Flags:
    --strict-markdown-secrets  Also scan inside Markdown fenced code blocks for
                               secrets (and merge markers).  Default behaviour
                               skips fenced blocks to reduce false positives.
    --allowlist FILE           Path(s) exempt from strict-mode fenced-block
                               scanning (may be repeated).
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Set, Tuple

# Directories to exclude
EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "source", "node_modules",
    "dist", "build", "__pycache__", "backups", "logs"
}

# File extensions to scan
INCLUDE_EXTENSIONS = {
    ".py", ".md", ".yml", ".yaml", ".toml", ".json",
    ".ini", ".cfg", ".ps1", ".bat", ".sh", ".txt"
}

# Patterns to detect — strict Git conflict tokens only (must be exactly 7 chars, optionally followed by a space and branch name)
MERGE_MARKERS = re.compile(r"^(<{7}|={7}|>{7})( .*)?$")
PLACEHOLDER_MARKERS = re.compile(
    r"TRUNCAT|TBD|TODO|FIXME|PLACEHOLDER|INSERT HERE|REPLACE ME|WIP|COMING SOON|CUT HERE",
    re.IGNORECASE
)
# Real secret patterns (not documentation placeholders)
SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9]{40,}"), "OpenAI API key"),  # Real keys are longer
    (re.compile(r'["\']?api[_-]?key["\']?\s*[:=]\s*["\'](?!sk-\.\.\.|your|example|test|dummy|xxx)[A-Za-z0-9]{20,}["\']', re.IGNORECASE), "Generic API key"),
    (re.compile(r"-----BEGIN (RSA |DSA )?PRIVATE KEY-----"), "Private key"),
    (re.compile(r"aws_secret_access_key\s*=\s*[A-Za-z0-9/+]{40}"), "AWS secret key"),
]


def should_scan_file(filepath: Path) -> bool:
    """Check if file should be scanned."""
    # Check if in excluded directory
    for part in filepath.parts:
        if part in EXCLUDE_DIRS:
            return False

    # Check file extension
    return filepath.suffix in INCLUDE_EXTENSIONS


def _fenced_line_set(lines: List[str]) -> set:
    """Return the set of 0-based line indices that are inside Markdown fenced code blocks."""
    inside = set()
    in_fence = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_fence:
                # closing fence — this line itself is still inside the block
                inside.add(idx)
                in_fence = False
            else:
                in_fence = True
                inside.add(idx)
        elif in_fence:
            inside.add(idx)
    return inside


def scan_file(
    filepath: Path,
    *,
    strict_markdown_secrets: bool = False,
    allowlisted_paths: Set[Path] | None = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Scan a file and return lists of issues found.

    Parameters
    ----------
    filepath:
        The file to scan.
    strict_markdown_secrets:
        When *True*, secrets (and merge markers) inside Markdown fenced code
        blocks are **not** skipped — they are reported just like any other
        match.  This catches real leaked keys pasted into documentation.
    allowlisted_paths:
        Set of resolved file paths that are exempt from strict-mode
        fenced-block scanning.  If *filepath* is in this set, fenced-block
        content is still skipped even in strict mode.
    """
    merge_issues = []
    placeholder_issues = []
    secret_issues = []

    if allowlisted_paths is None:
        allowlisted_paths = set()

    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        lines = content.split("\n")

        is_markdown = filepath.suffix == ".md"
        fenced = _fenced_line_set(lines) if is_markdown else set()

        # Determine whether fenced-block content should be skipped for this
        # particular file.  In strict mode the blocks are scanned *unless*
        # the file is on the allowlist.
        skip_fenced = True
        if strict_markdown_secrets and is_markdown:
            if filepath.resolve() not in allowlisted_paths:
                skip_fenced = False

        # Check for merge markers (strict: line must be exactly the marker)
        for i, line in enumerate(lines):
            if MERGE_MARKERS.match(line):
                # In Markdown, skip merge-marker look-alikes inside fenced blocks
                if is_markdown and i in fenced and skip_fenced:
                    continue
                merge_issues.append(f"{filepath}:{i + 1}: {line[:50]}")

        # Check for placeholders (but filter out common false positives)
        for i, line in enumerate(lines, 1):
            if PLACEHOLDER_MARKERS.search(line):
                # Skip if it's in a comment about the pattern itself
                if "scan" in line.lower() or "pattern" in line.lower():
                    continue
                # Skip if it's the word "placeholder" in legitimate context
                if "placeholder" in line.lower() and ("_path" in line.lower() or "voice" in line.lower()):
                    continue
                placeholder_issues.append(f"{filepath}:{i}: {line.strip()[:80]}")

        # Check for secrets
        for pattern, secret_type in SECRET_PATTERNS:
            for match in pattern.finditer(content):
                # Get line number (0-based)
                line_idx = content[:match.start()].count("\n")
                line_text = lines[line_idx] if line_idx < len(lines) else ""

                # Skip matches inside Markdown fenced code blocks
                if is_markdown and line_idx in fenced and skip_fenced:
                    continue

                # Skip lines that are clearly documenting the pattern (backtick-wrapped)
                if "`" in line_text and match.group() in line_text:
                    before_match = line_text[:line_text.index(match.group())]
                    if before_match.count("`") % 2 == 1:
                        continue

                secret_issues.append(f"{filepath}:{line_idx + 1}: Potential {secret_type}")

    except Exception as e:
        print(f"Error scanning {filepath}: {e}", file=sys.stderr)

    return merge_issues, placeholder_issues, secret_issues


def _parse_args(argv: list | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Security audit for Rex AI Assistant repository."
    )
    parser.add_argument(
        "--strict-markdown-secrets",
        action="store_true",
        default=False,
        help=(
            "Scan inside Markdown fenced code blocks for secrets and merge "
            "markers.  Default behaviour skips fenced blocks to reduce false "
            "positives."
        ),
    )
    parser.add_argument(
        "--allowlist",
        action="append",
        default=[],
        metavar="FILE",
        help=(
            "File path exempt from strict-mode fenced-block scanning.  May be "
            "repeated.  Paths are resolved relative to the repo root."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list | None = None):
    """Run security audit."""
    args = _parse_args(argv)

    repo_root = Path(__file__).parent.parent

    # Build the allowlist as a set of resolved Paths.
    allowlisted_paths: Set[Path] = set()
    for raw in args.allowlist:
        p = Path(raw)
        if not p.is_absolute():
            p = repo_root / p
        allowlisted_paths.add(p.resolve())

    strict = args.strict_markdown_secrets

    merge_findings = []
    placeholder_findings = []
    secret_findings = []
    files_scanned = 0

    print("Starting security audit...")
    print(f"Scanning: {repo_root}")
    print(f"Excluding: {', '.join(sorted(EXCLUDE_DIRS))}")
    if strict:
        print("Mode: --strict-markdown-secrets enabled")
        if allowlisted_paths:
            print(f"Allowlisted: {', '.join(str(p) for p in sorted(allowlisted_paths))}")
    print()

    # Scan all files
    for filepath in repo_root.rglob("*"):
        if filepath.is_file() and should_scan_file(filepath):
            files_scanned += 1
            merge, placeholder, secret = scan_file(
                filepath,
                strict_markdown_secrets=strict,
                allowlisted_paths=allowlisted_paths,
            )
            merge_findings.extend(merge)
            placeholder_findings.extend(placeholder)
            secret_findings.extend(secret)

    # Report results
    print(f"Scanned {files_scanned} files")
    print()

    print("=" * 70)
    print("MERGE CONFLICT MARKERS")
    print("=" * 70)
    if merge_findings:
        print(f"❌ FOUND {len(merge_findings)} issues:")
        for issue in merge_findings[:20]:
            print(f"  {issue}")
        if len(merge_findings) > 20:
            print(f"  ... and {len(merge_findings) - 20} more")
    else:
        print("✅ CLEAN - No merge markers found")
    print()

    print("=" * 70)
    print("PLACEHOLDER/INCOMPLETE CODE MARKERS")
    print("=" * 70)
    if placeholder_findings:
        print(f"⚠️  FOUND {len(placeholder_findings)} potential issues:")
        for issue in placeholder_findings[:30]:
            print(f"  {issue}")
        if len(placeholder_findings) > 30:
            print(f"  ... and {len(placeholder_findings) - 30} more")
    else:
        print("✅ CLEAN - No placeholder markers found")
    print()

    print("=" * 70)
    print("EXPOSED SECRETS")
    print("=" * 70)
    if secret_findings:
        print(f"🚨 FOUND {len(secret_findings)} potential secrets:")
        for issue in secret_findings:
            print(f"  {issue}")
        return 1
    else:
        print("✅ CLEAN - No exposed secrets found")
    print()

    # Return exit code
    if secret_findings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
