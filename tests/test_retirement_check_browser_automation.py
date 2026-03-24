"""Retirement confirmation for rex/browser_automation.py (US-P7-011).

Status: RETIRED
  rex/browser_automation.py has been deleted.
  Core browser primitives (BrowserSession, run_browser_script) moved to
  rex/openclaw/browser_core.py.
  rex/openclaw/browser_bridge.py was rewritten to use browser_core directly.

This test acts as a regression guard to ensure the module stays retired
and no new callers appear.
"""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent
REX_PKG = REPO_ROOT / "rex"

EXEMPT_PATHS: set[str] = {
    "rex/contracts/browser.py",
}


def _imports_browser_automation(path: pathlib.Path) -> bool:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return False
    if "browser_automation" not in source:
        return False
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and "browser_automation" in (node.module or ""):
            return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "browser_automation" in alias.name:
                    return True
    return False


def _find_active_importers() -> set[str]:
    """Check ALL files for imports of rex.browser_automation."""
    importers = set()
    for py_file in REPO_ROOT.glob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        if _imports_browser_automation(py_file):
            importers.add(py_file.name)
    for py_file in REX_PKG.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        if any(rel == e or rel.endswith(e) for e in EXEMPT_PATHS):
            continue
        if "__pycache__" in rel:
            continue
        if _imports_browser_automation(py_file):
            importers.add(rel)
    tests_dir = REPO_ROOT / "tests"
    for py_file in tests_dir.glob("*.py"):
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        if "__pycache__" in rel:
            continue
        if _imports_browser_automation(py_file):
            importers.add(rel)
    return importers


class TestBrowserAutomationRetired:
    """Regression guard — rex/browser_automation.py must stay retired."""

    def test_module_deleted(self):
        """rex/browser_automation.py no longer exists in the codebase."""
        assert not (REX_PKG / "browser_automation.py").exists(), (
            "rex/browser_automation.py was re-created! "
            "Use rex.openclaw.browser_bridge.BrowserBridge and "
            "rex.openclaw.browser_core.BrowserSession instead."
        )

    def test_no_active_importers(self):
        """No file imports from rex.browser_automation (module is retired)."""
        active = _find_active_importers()
        assert not active, (
            f"New callers of the retired rex.browser_automation found: {active}\n"
            "Use rex.openclaw.browser_bridge.BrowserBridge or "
            "rex.openclaw.browser_core.BrowserSession instead."
        )
