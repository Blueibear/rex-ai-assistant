"""Pre-retirement check for rex/tool_registry.py (US-P7-005).

Verdict: NOT SAFE TO RETIRE
  Active importers: rex/__init__.py, rex/cli.py, rex/planner.py
"""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent
REX_PKG = REPO_ROOT / "rex"

KNOWN_BLOCKERS = {
    "rex/__init__.py",
    "rex/cli.py",
    "rex/planner.py",
}

EXEMPT_PATHS = {"rex/tool_registry.py", "rex/contracts/tools.py"}


def _imports_tool_registry(path: pathlib.Path) -> bool:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return False
    if "tool_registry" not in source:
        return False
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and "tool_registry" in (node.module or ""):
            return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "tool_registry" in alias.name:
                    return True
    return False


def _find_active_importers() -> set[str]:
    importers = set()
    for py_file in REX_PKG.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        if any(rel == e or rel.endswith(e) for e in EXEMPT_PATHS):
            continue
        if "__pycache__" in rel or "openclaw" in rel:
            continue
        if _imports_tool_registry(py_file):
            importers.add(rel)
    return importers


class TestToolRegistryRetirementCheck:
    def test_module_exists(self):
        assert (REX_PKG / "tool_registry.py").exists()

    def test_has_openclaw_replace_marker(self):
        assert "OPENCLAW-REPLACE" in (REX_PKG / "tool_registry.py").read_text(encoding="utf-8")

    def test_known_blockers_still_present(self):
        active = _find_active_importers()
        migrated = KNOWN_BLOCKERS - active
        assert KNOWN_BLOCKERS & active == KNOWN_BLOCKERS, (
            f"Migrated (update KNOWN_BLOCKERS): {migrated}"
        )

    def test_no_unexpected_new_importers(self):
        active = _find_active_importers()
        unexpected = active - KNOWN_BLOCKERS
        assert not unexpected, f"New importers: {unexpected}"

    def test_retirement_verdict_not_safe(self):
        active = _find_active_importers()
        assert active & KNOWN_BLOCKERS, "All blockers migrated — safe to retire!"
