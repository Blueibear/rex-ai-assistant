#!/usr/bin/env python3
"""Validate the skipped-test inventory against executable pytest skip sites.

US-038 requires every current skip site to carry an explicit action and a
non-circular follow-up when work remains. The checker uses only the standard
library so it can run in the lightweight CI lint job.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

INVENTORY_PATH = Path("docs/testing/SKIPPED-TESTS-INVENTORY.md")
TESTS_ROOT = Path("tests")
VALID_ACTIONS = {"keep", "fix", "replace", "archive"}
VALID_CLASSIFICATIONS = {
    "optional-dep-skip",
    "platform-skip",
    "retired-surface-skip",
    "temporary-bug-skip",
}
FOLLOW_UP_RE = re.compile(r"^US-\d+$")


@dataclass(frozen=True, order=True)
class SkipSite:
    path: str
    line: int
    skip_type: str
    reason: str


@dataclass(frozen=True, order=True)
class InventoryRow:
    path: str
    line: int
    skip_type: str
    reason: str
    classification: str
    action: str
    follow_up: str

    @property
    def site(self) -> SkipSite:
        return SkipSite(self.path, self.line, self.skip_type, self.reason)


def _dotted_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _reason_node(call: ast.Call, skip_type: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == "reason":
            return keyword.value
    if skip_type == "skipif" and len(call.args) >= 2:
        return call.args[1]
    if skip_type in {"skip", "pytest.skip"} and call.args:
        return call.args[0]
    return None


def _reason_text(source: str, node: ast.AST | None) -> str:
    if node is None:
        return "<reason missing>"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return (ast.get_source_segment(source, node) or ast.unparse(node)).strip()


def _skip_type(call: ast.Call) -> str | None:
    name = _dotted_name(call.func)
    if name == "pytest.skip":
        return "pytest.skip"
    if name == "skip":
        return "skip"
    if name == "pytest.mark.skip" or name.endswith(".mark.skip"):
        return "skip"
    if name == "pytest.mark.skipif" or name.endswith(".mark.skipif"):
        return "skipif"
    return None


def scan_skip_sites(tests_root: Path = TESTS_ROOT) -> list[SkipSite]:
    """Return all executable pytest skip calls/decorators under *tests_root*."""
    sites: list[SkipSite] = []
    for path in sorted(tests_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            skip_type = _skip_type(node)
            if skip_type is None:
                continue
            reason = _reason_text(source, _reason_node(node, skip_type))
            sites.append(SkipSite(path.as_posix(), node.lineno, skip_type, reason))
    return sorted(sites)


def parse_inventory(path: Path = INVENTORY_PATH) -> list[InventoryRow]:
    """Parse the seven-column Markdown inventory table."""
    rows: list[InventoryRow] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("| `tests/"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 7:
            raise ValueError(
                f"{path}:{line_number}: expected 7 inventory columns, found {len(parts)}"
            )
        file_text, line_text, skip_type, reason, classification, action, follow_up = parts
        rows.append(
            InventoryRow(
                path=file_text.strip("`"),
                line=int(line_text),
                skip_type=skip_type.strip("`"),
                reason=reason,
                classification=classification.strip("`"),
                action=action.strip("`"),
                follow_up=follow_up.strip("`"),
            )
        )
    return sorted(rows)


def _validate_row_policy(row: InventoryRow) -> list[str]:
    """Return classification/action/follow-up errors for one inventory row."""
    errors: list[str] = []
    if row.classification not in VALID_CLASSIFICATIONS:
        return [f"invalid classification at {row.path}:{row.line}: {row.classification!r}"]
    if row.action not in VALID_ACTIONS:
        return [f"invalid action at {row.path}:{row.line}: {row.action!r}"]

    if row.classification in {"optional-dep-skip", "platform-skip"}:
        if row.action != "keep":
            errors.append(f"permanent guard must use keep at {row.path}:{row.line}")
        if not row.follow_up.startswith("permanent:"):
            errors.append(f"kept guard needs permanent rationale at {row.path}:{row.line}")
    elif row.classification == "retired-surface-skip":
        if row.action != "archive" or row.follow_up != "US-039":
            errors.append(f"retired surface must archive under US-039 at {row.path}:{row.line}")
    elif row.classification == "temporary-bug-skip":
        if row.action not in {"fix", "replace"}:
            errors.append(f"temporary bug must use fix/replace at {row.path}:{row.line}")
        if not FOLLOW_UP_RE.fullmatch(row.follow_up) or row.follow_up == "US-038":
            errors.append(f"temporary bug needs non-circular story ID at {row.path}:{row.line}")
    return errors


def validate_inventory(
    actual_sites: list[SkipSite], inventory_rows: list[InventoryRow]
) -> list[str]:
    """Return actionable inventory validation errors."""
    errors: list[str] = []
    actual_by_location = {(site.path, site.line, site.skip_type): site for site in actual_sites}
    inventory_by_location = {(row.path, row.line, row.skip_type): row for row in inventory_rows}

    missing = sorted(actual_by_location.keys() - inventory_by_location.keys())
    stale = sorted(inventory_by_location.keys() - actual_by_location.keys())
    for path, line, skip_type in missing:
        errors.append(f"missing inventory row: {path}:{line} ({skip_type})")
    for path, line, skip_type in stale:
        errors.append(f"stale inventory row: {path}:{line} ({skip_type})")

    for location in sorted(actual_by_location.keys() & inventory_by_location.keys()):
        site = actual_by_location[location]
        row = inventory_by_location[location]
        if row.reason != site.reason:
            errors.append(
                f"reason drift at {site.path}:{site.line}: "
                f"inventory={row.reason!r}, source={site.reason!r}"
            )
        errors.extend(_validate_row_policy(row))
    return errors


def main() -> int:
    try:
        sites = scan_skip_sites()
        rows = parse_inventory()
    except (OSError, SyntaxError, ValueError) as exc:
        print(f"ERROR: unable to validate skip inventory: {exc}", file=sys.stderr)
        return 2
    errors = validate_inventory(sites, rows)
    if errors:
        print("ERROR: skipped-test inventory is out of date:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"OK: {len(rows)} executable skip sites are current and have explicit actions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
