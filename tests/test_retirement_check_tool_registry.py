"""Permanent guard: rex/tool_registry.py was retired in US-P7-006.

This file confirms the module no longer exists and no rex/ code imports from it.
Logic was relocated to rex/openclaw/tool_registry.py.
"""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent
REX_PKG = REPO_ROOT / "rex"

EXEMPT_PATHS = {"rex/openclaw/tool_registry.py", "rex/contracts/tools.py"}


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
            # Only flag imports from rex.tool_registry (the retired path)
            if node.module and node.module.startswith("rex.tool_registry"):
                return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "rex.tool_registry":
                    return True
    return False


def _find_active_importers() -> set[str]:
    importers = set()
    for py_file in REX_PKG.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        if any(rel == e or rel.endswith(e) for e in EXEMPT_PATHS):
            continue
        if "__pycache__" in rel:
            continue
        if _imports_tool_registry(py_file):
            importers.add(rel)
    return importers


class TestToolRegistryRetired:
    def test_module_does_not_exist(self):
        assert not (REX_PKG / "tool_registry.py").exists(), (
            "rex/tool_registry.py was re-introduced — it was retired in US-P7-006"
        )

    def test_no_rex_importers(self):
        active = _find_active_importers()
        assert not active, f"Files still import rex.tool_registry: {active}"
