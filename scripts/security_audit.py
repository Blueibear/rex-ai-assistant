#!/usr/bin/env python3
"""Security audit script for Rex AI Assistant repository.

Scans source files for:
- Merge conflict markers
- Placeholder/incomplete code markers
- Exposed secrets and API keys

Excludes: .git, .venv, venv, source, node_modules, dist, build, __pycache__, backups, logs
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

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

# Patterns to detect
MERGE_MARKERS = re.compile(r"^(<{7}|={7}|>{7})", re.MULTILINE)
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


def scan_file(filepath: Path) -> Tuple[List[str], List[str], List[str]]:
    """Scan a file and return lists of issues found."""
    merge_issues = []
    placeholder_issues = []
    secret_issues = []

    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        lines = content.split("\n")

        # Check for merge markers
        for i, line in enumerate(lines, 1):
            if MERGE_MARKERS.match(line):
                merge_issues.append(f"{filepath}:{i}: {line[:50]}")

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
                # Get line number
                line_num = content[:match.start()].count("\n") + 1
                secret_issues.append(f"{filepath}:{line_num}: Potential {secret_type}")

    except Exception as e:
        print(f"Error scanning {filepath}: {e}", file=sys.stderr)

    return merge_issues, placeholder_issues, secret_issues


def main():
    """Run security audit."""
    repo_root = Path(__file__).parent.parent

    merge_findings = []
    placeholder_findings = []
    secret_findings = []
    files_scanned = 0

    print("Starting security audit...")
    print(f"Scanning: {repo_root}")
    print(f"Excluding: {', '.join(sorted(EXCLUDE_DIRS))}")
    print()

    # Scan all files
    for filepath in repo_root.rglob("*"):
        if filepath.is_file() and should_scan_file(filepath):
            files_scanned += 1
            merge, placeholder, secret = scan_file(filepath)
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
