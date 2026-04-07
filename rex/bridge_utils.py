"""Utilities shared by all rex bridge scripts."""

import os
import sys
from pathlib import Path


def resolve_python() -> str:
    """Return the absolute path to the active venv Python interpreter.

    Resolution order:
    1. If running inside a virtualenv, derive the interpreter path from the
       ``VIRTUAL_ENV`` environment variable.
    2. Fall back to ``sys.executable`` (works when the current process itself
       was started with the right interpreter, e.g. in CI or when activated).

    Platform behaviour
    ------------------
    - Windows: ``<venv>\\Scripts\\python.exe``
    - macOS / Linux: ``<venv>/bin/python``
    """
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        venv_path = Path(venv)
        if sys.platform == "win32":
            candidate = venv_path / "Scripts" / "python.exe"
        else:
            candidate = venv_path / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    return sys.executable


def repo_root() -> Path:
    """Return the absolute path to the repository root.

    The root is the directory that contains ``pyproject.toml``.  We walk up
    from this file's location until we find it.
    """
    current = Path(__file__).resolve().parent
    while True:
        if (current / "pyproject.toml").exists():
            return current
        parent = current.parent
        if parent == current:
            # Reached filesystem root without finding pyproject.toml
            raise RuntimeError(
                "Could not locate repository root (pyproject.toml not found)"
            )
        current = parent
