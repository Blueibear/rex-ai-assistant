"""Pre-retirement check for rex/messaging_backends/ and rex/messaging_service.py (US-P7-015).

Verdict: NOT SAFE TO RETIRE
  Active importers of messaging_service: rex/__init__.py, rex/cli.py, rex/services.py
  Migrated: rex/notification.py (_send_to_sms converted to logging stub, iter 88)
"""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent
REX_PKG = REPO_ROOT / "rex"

KNOWN_BLOCKERS = {
    "rex/__init__.py",
    "rex/cli.py",
    "rex/services.py",
}

EXEMPT_PATHS = {
    "rex/messaging_service.py",
    "rex/messaging_backends/",
    "rex/contracts/messaging.py",
}


def _imports_messaging(path: pathlib.Path) -> bool:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return False
    if "messaging_service" not in source and "MessagingService" not in source:
        return False
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if "messaging_service" in module:
                return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "messaging_service" in alias.name:
                    return True
    return "messaging_service" in source


def _find_active_importers() -> set[str]:
    importers = set()
    for py_file in REX_PKG.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        if any(rel.startswith(e) or rel == e for e in EXEMPT_PATHS):
            continue
        if any(part in rel for part in ("__pycache__", "openclaw")):
            continue
        if _imports_messaging(py_file):
            importers.add(rel)
    return importers


class TestMessagingRetirementCheck:
    def test_messaging_service_module_exists(self):
        assert (REX_PKG / "messaging_service.py").exists()

    def test_has_openclaw_replace_marker(self):
        assert "OPENCLAW-REPLACE" in (REX_PKG / "messaging_service.py").read_text(encoding="utf-8")

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
