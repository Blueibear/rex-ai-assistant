"""Pre-retirement check for rex/plugin_loader.py (US-P7-003).

Audits whether rex.plugin_loader can be safely retired.

Verdict: NOT SAFE TO RETIRE
  1 production file still imports from rex.plugin_loader:
  - voice_loop.py (root) — imports load_plugins at line 98

Known blockers:
  - voice_loop.py  — imports load_plugins from rex.plugin_loader directly

Migration path:
  - voice_loop.py should switch to rex.plugins.load_plugins (already used by rex/cli.py)
"""

from __future__ import annotations

import ast
import pathlib


REPO_ROOT = pathlib.Path(__file__).parent.parent
REX_PKG = REPO_ROOT / "rex"

KNOWN_BLOCKERS = {
    "voice_loop.py",  # root-level, imports load_plugins from rex.plugin_loader
}

EXEMPT_PATHS = {
    "rex/plugin_loader.py",
    "rex/contracts/plugins.py",
}


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
    # Check root-level py files
    for py_file in REPO_ROOT.glob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        rel = py_file.name
        if _imports_plugin_loader(py_file):
            importers.add(rel)
    # Check rex/ package
    for py_file in REX_PKG.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        if any(rel == e or rel.endswith(e) for e in EXEMPT_PATHS):
            continue
        if "__pycache__" in rel:
            continue
        if _imports_plugin_loader(py_file):
            importers.add(rel)
    return importers


class TestPluginLoaderRetirementCheck:
    """Pre-retirement audit for rex/plugin_loader.py."""

    def test_plugin_loader_module_exists(self):
        """rex/plugin_loader.py still exists — not prematurely removed."""
        assert (REX_PKG / "plugin_loader.py").exists()

    def test_plugin_loader_has_openclaw_replace_marker(self):
        """rex/plugin_loader.py carries the OPENCLAW-REPLACE marker."""
        content = (REX_PKG / "plugin_loader.py").read_text(encoding="utf-8")
        assert "OPENCLAW-REPLACE" in content

    def test_known_blockers_still_present(self):
        """Known blockers still import rex.plugin_loader."""
        active = _find_active_importers()
        still_blocking = KNOWN_BLOCKERS & active
        migrated = KNOWN_BLOCKERS - active
        assert still_blocking == KNOWN_BLOCKERS, (
            f"Some blockers were migrated (update KNOWN_BLOCKERS): {migrated}"
        )

    def test_no_unexpected_new_importers(self):
        """No new files have started importing rex.plugin_loader."""
        active = _find_active_importers()
        unexpected = active - KNOWN_BLOCKERS
        assert not unexpected, (
            f"New unexpected importers of rex.plugin_loader: {unexpected}\n"
            "Add them to KNOWN_BLOCKERS or migrate them."
        )

    def test_retirement_verdict_not_safe(self):
        """Confirm retirement verdict: NOT SAFE (voice_loop.py blocker remains)."""
        active = _find_active_importers()
        remaining = active & KNOWN_BLOCKERS
        assert remaining, (
            "All known blockers are migrated — rex/plugin_loader.py may be safe to retire!\n"
            "Verify rex.plugins.load_plugins is the replacement, then proceed to US-P7-004."
        )
