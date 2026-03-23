"""Pre-retirement check for rex/dashboard_store.py and rex/dashboard/ (US-P7-013).

Verdict: NOT SAFE TO RETIRE
  Active importers of dashboard_store: rex/digest_job.py, rex/gui_app.py,
  rex/messaging_backends/inbound_store.py, rex/notification.py

  Migrated callers:
  - rex/health.py — check_dashboard_db removed (health checks are now
    dashboard-independent as part of the OpenClaw migration)
  - rex/retention.py — setup_dashboard_cleanup_job converted to no-op
    (OpenClaw will manage dashboard state retention after store retirement)
"""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent
REX_PKG = REPO_ROOT / "rex"

KNOWN_BLOCKERS = {
    "rex/digest_job.py",
    "rex/gui_app.py",
    "rex/messaging_backends/inbound_store.py",
    "rex/notification.py",
}

EXEMPT_PATHS = {
    "rex/dashboard_store.py",
    "rex/dashboard/",
    "rex/contracts/dashboard.py",
}


def _imports_dashboard(path: pathlib.Path) -> bool:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return False
    if "dashboard" not in source:
        return False
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if "dashboard_store" in module or (
                module.startswith("rex.dashboard") and not module.startswith("rex.dashboard.")
            ):
                return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "dashboard_store" in alias.name:
                    return True
    # Also check string-based references (simpler heuristic)
    return "dashboard_store" in source


def _find_active_importers() -> set[str]:
    importers = set()
    for py_file in REX_PKG.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        if any(rel.startswith(e) or rel == e for e in EXEMPT_PATHS):
            continue
        if any(part in rel for part in ("__pycache__", "openclaw")):
            continue
        if _imports_dashboard(py_file):
            importers.add(rel)
    return importers


class TestDashboardRetirementCheck:
    def test_dashboard_store_module_exists(self):
        assert (REX_PKG / "dashboard_store.py").exists()

    def test_has_openclaw_replace_marker(self):
        assert "OPENCLAW-REPLACE" in (REX_PKG / "dashboard_store.py").read_text(encoding="utf-8")

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
