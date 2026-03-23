"""Retirement confirmation for rex/executor.py (US-P7-010).

Status: RETIRED
  rex/executor.py has been deleted.
  rex/cli.py was migrated to rex.openclaw.workflow_bridge.WorkflowBridge.

This test acts as a regression guard to ensure the module stays retired
and no new callers appear.
"""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent
REX_PKG = REPO_ROOT / "rex"

EXEMPT_PATHS: set[str] = set()


def _imports_executor(path: pathlib.Path) -> bool:
    """Return True if the file imports from rex.executor."""
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return False
    if "executor" not in source:
        return False
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return True

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "rex.executor" or module.endswith(".executor"):
                return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("rex.executor",):
                    return True
    return False


def _find_active_importers() -> set[str]:
    """Return relative paths of files that import rex.executor."""
    importers = set()
    for py_file in REPO_ROOT.glob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        if _imports_executor(py_file):
            importers.add(py_file.name)
    for py_file in REX_PKG.rglob("*.py"):
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        if any(rel == e or rel.endswith(e) for e in EXEMPT_PATHS):
            continue
        if "__pycache__" in rel:
            continue
        if _imports_executor(py_file):
            importers.add(rel)
    # Also scan tests/ to catch lingering test-file references
    tests_dir = REPO_ROOT / "tests"
    for py_file in tests_dir.glob("*.py"):
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        if "__pycache__" in rel:
            continue
        if _imports_executor(py_file):
            importers.add(rel)
    return importers


class TestExecutorRetired:
    """Regression guard — rex/executor.py must stay retired."""

    def test_executor_module_deleted(self):
        """rex/executor.py no longer exists in the codebase."""
        assert not (REX_PKG / "executor.py").exists(), (
            "rex/executor.py was re-created! "
            "Use rex.openclaw.workflow_bridge.WorkflowBridge instead."
        )

    def test_no_active_importers(self):
        """No file imports from rex.executor (module is retired)."""
        active = _find_active_importers()
        assert not active, (
            f"New callers of the retired rex.executor found: {active}\n"
            "Use rex.openclaw.workflow_bridge.WorkflowBridge instead."
        )
