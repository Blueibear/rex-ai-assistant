"""Permanent guard: rex/tool_router.py was retired in US-P7-008.

This file confirms the module no longer exists and no rex/ code imports from it.
"""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent
REX_PKG = REPO_ROOT / "rex"

EXEMPT_PATHS = {"rex/contracts/tool_routing.py", "rex/openclaw/tool_executor.py"}


def _imports_tool_router(path: pathlib.Path) -> bool:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return False
    if "tool_router" not in source:
        return False
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and "tool_router" in (node.module or ""):
            return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "tool_router" in alias.name:
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
        if _imports_tool_router(py_file):
            importers.add(rel)
    return importers


class TestToolRouterRetired:
    def test_module_does_not_exist(self):
        assert not (
            REX_PKG / "tool_router.py"
        ).exists(), "rex/tool_router.py was re-introduced — it was retired in US-P7-008"

    def test_no_rex_importers(self):
        active = _find_active_importers()
        assert not active, f"Files still import rex.tool_router: {active}"
