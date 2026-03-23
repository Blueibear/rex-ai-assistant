"""Pre-retirement check for rex/browser_automation.py (US-P7-011).

Verdict: NOT SAFE TO RETIRE
  rex/openclaw/browser_bridge.py imports BrowserAutomationService and
  get_browser_service from rex.browser_automation to delegate to it.
  The bridge must be rewritten to not depend on this module before retirement.

Known blockers:
  - rex/openclaw/browser_bridge.py — BrowserBridge delegates to BrowserAutomationService
"""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent
REX_PKG = REPO_ROOT / "rex"

# All files (including openclaw) that still import from rex.browser_automation
KNOWN_BLOCKERS = {
    "rex/openclaw/browser_bridge.py",
}

EXEMPT_PATHS = {
    "rex/browser_automation.py",
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
    """Check ALL rex/ files including openclaw (bridge still depends on legacy module)."""
    importers = set()
    for py_file in REX_PKG.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        if any(rel == e or rel.endswith(e) for e in EXEMPT_PATHS):
            continue
        if "__pycache__" in rel:
            continue
        if _imports_browser_automation(py_file):
            importers.add(rel)
    return importers


class TestBrowserAutomationRetirementCheck:
    def test_module_exists(self):
        """rex/browser_automation.py still exists — not prematurely removed."""
        assert (REX_PKG / "browser_automation.py").exists()

    def test_has_openclaw_replace_marker(self):
        assert "OPENCLAW-REPLACE" in (REX_PKG / "browser_automation.py").read_text(encoding="utf-8")

    def test_known_blockers_still_present(self):
        """browser_bridge.py still imports from browser_automation."""
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
        """Confirm retirement verdict: NOT SAFE (browser_bridge.py depends on it)."""
        active = _find_active_importers()
        assert active & KNOWN_BLOCKERS, (
            "browser_bridge.py has been migrated — browser_automation.py may now be safe to retire!"
        )
