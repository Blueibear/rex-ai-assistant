"""Pre-retirement check for rex/browser_automation.py (US-P7-011).

Verdict: SAFE TO RETIRE
  No production files import from rex.browser_automation.
  Only build artifacts and scripts reference it.
"""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent
REX_PKG = REPO_ROOT / "rex"

KNOWN_BLOCKERS: set[str] = set()  # No active importers

EXEMPT_PATHS = {
    "rex/browser_automation.py",
    "rex/contracts/browser.py",
    "build/",  # build artifacts
    "scripts/",  # dev scripts
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
    importers = set()
    # Check rex/ package (excluding openclaw adapters, contracts, the module itself, builds)
    for py_file in REX_PKG.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        if any(part in rel for part in ("__pycache__", "openclaw", "contracts", "browser_automation.py")):
            continue
        if _imports_browser_automation(py_file):
            importers.add(rel)
    return importers


class TestBrowserAutomationRetirementCheck:
    def test_module_exists(self):
        """rex/browser_automation.py still exists (not yet removed)."""
        assert (REX_PKG / "browser_automation.py").exists()

    def test_has_openclaw_replace_marker(self):
        assert "OPENCLAW-REPLACE" in (REX_PKG / "browser_automation.py").read_text(encoding="utf-8")

    def test_no_active_importers(self):
        """No production rex/ files import browser_automation — safe to retire."""
        active = _find_active_importers()
        assert not active, (
            f"Unexpected active importers found: {active}\n"
            "Add them to KNOWN_BLOCKERS before retiring."
        )

    def test_retirement_verdict_safe(self):
        """Confirm retirement verdict: SAFE TO RETIRE (no blockers)."""
        active = _find_active_importers()
        assert not active, f"Blockers remain: {active}"
