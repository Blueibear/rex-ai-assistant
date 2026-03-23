"""Permanent guard: rex/event_bus.py was retired in US-P7-002.

This file confirms the module no longer exists and no rex/ code imports from it.
Logic was relocated to rex/openclaw/event_bus.py.
"""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent
REX_PKG = REPO_ROOT / "rex"

EXEMPT_PATHS = {
    "rex/openclaw/event_bus.py",
    "rex/contracts/event_bus.py",
}


def _imports_legacy_event_bus(path: pathlib.Path) -> bool:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return False
    if "event_bus" not in source:
        return False
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            # Flag only imports from the retired rex.event_bus path
            if module == "rex.event_bus":
                return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "rex.event_bus":
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
        if _imports_legacy_event_bus(py_file):
            importers.add(rel)
    return importers


class TestEventBusRetired:
    def test_module_does_not_exist(self):
        assert not (REX_PKG / "event_bus.py").exists(), (
            "rex/event_bus.py was re-introduced — it was retired in US-P7-002"
        )

    def test_no_rex_importers(self):
        active = _find_active_importers()
        assert not active, f"Files still import rex.event_bus: {active}"
