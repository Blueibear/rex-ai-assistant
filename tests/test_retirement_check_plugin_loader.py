"""Retirement confirmation for rex/plugin_loader.py (US-P7-004).

Status: RETIRED
  rex/plugin_loader.py has been deleted.
  voice_loop.py (root) was migrated to rex.plugins.load_plugins.

This test acts as a regression guard to ensure the module stays retired
and no new callers appear.
"""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent
REX_PKG = REPO_ROOT / "rex"

EXEMPT_PATHS: set[str] = set()


def _imports_plugin_loader(path: pathlib.Path) -> bool:
    """Return True if the file imports from rex.plugin_loader."""
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return False
    if "plugin_loader" not in source:
        return False
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return True

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if "plugin_loader" in module:
                return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "plugin_loader" in alias.name:
                    return True
    return False


def _find_active_importers() -> set[str]:
    """Return relative paths of files that import rex.plugin_loader."""
    importers = set()
    for py_file in REPO_ROOT.glob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        if _imports_plugin_loader(py_file):
            importers.add(py_file.name)
    for py_file in REX_PKG.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        if any(rel == e or rel.endswith(e) for e in EXEMPT_PATHS):
            continue
        if "__pycache__" in rel:
            continue
        if _imports_plugin_loader(py_file):
            importers.add(rel)
    return importers


class TestPluginLoaderRetired:
    """Regression guard — rex/plugin_loader.py must stay retired."""

    def test_plugin_loader_module_deleted(self):
        """rex/plugin_loader.py no longer exists in the codebase."""
        assert not (REX_PKG / "plugin_loader.py").exists(), (
            "rex/plugin_loader.py was re-created! " "Use rex.plugins.load_plugins instead."
        )

    def test_no_active_importers(self):
        """No file imports from rex.plugin_loader (module is retired)."""
        active = _find_active_importers()
        assert not active, (
            f"New callers of the retired rex.plugin_loader found: {active}\n"
            "Use rex.plugins.load_plugins instead."
        )
