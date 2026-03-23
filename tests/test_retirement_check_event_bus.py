"""Pre-retirement check for rex/event_bus.py (US-P7-001).

Audits whether rex.event_bus can be safely retired.

Verdict: NOT SAFE TO RETIRE
  7 production files still import from rex.event_bus.
  They must be migrated to EventBridge before retirement.

Known blockers:
  - rex/__init__.py                      — re-exports EventBus as public API
  - rex/calendar_service.py             — uses EventBus for calendar.*  events
  - rex/email_service.py                — uses EventBus for email.*    events
  - rex/event_triggers.py               — uses EventBus/EventBridge together
  - rex/integrations.py                 — uses get_event_bus()
  - rex/integrations/_setup.py         — uses get_event_bus() and EventBridge
  - rex/openclaw/ha_event_subscriber.py — imports EventBus for type annotation
  - rex/services.py                     — creates EventBus and passes to services
"""

from __future__ import annotations

import ast
import pathlib


REPO_ROOT = pathlib.Path(__file__).parent.parent
REX_PKG = REPO_ROOT / "rex"

# Files known to still import from rex.event_bus (blockers for retirement)
KNOWN_BLOCKERS = {
    "rex/__init__.py",
    "rex/calendar_service.py",
    "rex/email_service.py",
    "rex/event_triggers.py",
    "rex/integrations.py",
    "rex/integrations/_setup.py",
    "rex/openclaw/ha_event_subscriber.py",  # imports EventBus for type annotation only
    "rex/services.py",
}

# Files that are exempt from the check (the module itself, openclaw adapters, contracts)
EXEMPT_SUFFIXES = (
    "rex/event_bus.py",
    "rex/contracts/event_bus.py",
    "rex/openclaw/event_bridge.py",
)


def _imports_event_bus(path: pathlib.Path) -> bool:
    """Return True if the file contains any import from rex.event_bus."""
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return False
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return source.count("event_bus") > 0

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if "event_bus" in module:
                return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "event_bus" in alias.name:
                    return True
    return False


def _find_active_importers() -> set[str]:
    """Return relative paths of files that import rex.event_bus (excluding exempts)."""
    importers = set()
    for py_file in REX_PKG.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        if any(rel.endswith(exempt) or rel == exempt for exempt in EXEMPT_SUFFIXES):
            continue
        if "__pycache__" in rel:
            continue
        if _imports_event_bus(py_file):
            importers.add(rel)
    return importers


class TestEventBusRetirementCheck:
    """Pre-retirement audit for rex/event_bus.py."""

    def test_event_bus_module_exists(self):
        """rex/event_bus.py still exists — not prematurely removed."""
        assert (REX_PKG / "event_bus.py").exists(), (
            "rex/event_bus.py was removed before all callers were migrated"
        )

    def test_event_bus_has_openclaw_replace_marker(self):
        """rex/event_bus.py carries the OPENCLAW-REPLACE marker."""
        content = (REX_PKG / "event_bus.py").read_text(encoding="utf-8")
        assert "OPENCLAW-REPLACE" in content

    def test_known_blockers_still_present(self):
        """All known blockers still import rex.event_bus (tracks migration progress)."""
        active = _find_active_importers()
        still_blocking = KNOWN_BLOCKERS & active
        migrated = KNOWN_BLOCKERS - active
        # Report progress — this test passes as long as the set is accurate
        # (update KNOWN_BLOCKERS as files get migrated)
        assert still_blocking == KNOWN_BLOCKERS, (
            f"Some blockers were migrated (update KNOWN_BLOCKERS): {migrated}"
        )

    def test_no_unexpected_new_importers(self):
        """No new files have been added that import rex.event_bus."""
        active = _find_active_importers()
        unexpected = active - KNOWN_BLOCKERS
        assert not unexpected, (
            f"New unexpected importers of rex.event_bus found: {unexpected}\n"
            "Add them to KNOWN_BLOCKERS or migrate them."
        )

    def test_retirement_verdict_not_safe(self):
        """Confirm retirement verdict: NOT SAFE (blockers remain)."""
        active = _find_active_importers()
        remaining = active & KNOWN_BLOCKERS
        assert remaining, (
            "All known blockers are migrated — rex/event_bus.py is now safe to retire!\n"
            "Proceed to US-P7-002."
        )
