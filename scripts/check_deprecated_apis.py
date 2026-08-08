#!/usr/bin/env python3
"""Fail CI if deprecated datetime/asyncio call patterns return.

The guard parses tracked Python source with the stdlib AST so references inside
comments, docs, and string-based regression tests do not create false positives.
Historical code under archived/ is intentionally excluded.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class Finding:
    path: Path
    line: int
    column: int
    api: str
    replacement: str


_BANNED = {
    "datetime.utcnow": "use timezone-aware datetime.now(UTC)",
    "asyncio.get_event_loop": "use get_running_loop() in async code or an explicit loop lifecycle",
}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def scan_file(path: Path) -> list[Finding]:
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        api = _call_name(node.func)
        replacement = _BANNED.get(api)
        if replacement is None:
            continue
        findings.append(
            Finding(
                path=path,
                line=node.lineno,
                column=node.col_offset + 1,
                api=api,
                replacement=replacement,
            )
        )
    return sorted(findings)


def _is_archived(path: Path, root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return bool(relative.parts and relative.parts[0].lower() == "archived")


def scan_paths(paths: Iterable[Path], *, root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        if _is_archived(path, root):
            continue
        findings.extend(scan_file(path))
    return sorted(findings)


def tracked_python_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "*.py"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        findings = scan_paths(tracked_python_files(root), root=root)
    except (OSError, SyntaxError, subprocess.CalledProcessError) as exc:
        print(f"Deprecated-API guard failed closed: {exc}", file=sys.stderr)
        return 2

    if findings:
        print("Deprecated API calls are not allowed outside archived/:", file=sys.stderr)
        for finding in findings:
            try:
                display = finding.path.relative_to(root)
            except ValueError:
                display = finding.path
            print(
                f"  {display}:{finding.line}:{finding.column}: {finding.api}() - {finding.replacement}",
                file=sys.stderr,
            )
        return 1

    print(
        "OK: no banned datetime.utcnow() or asyncio.get_event_loop() calls in tracked current Python."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
