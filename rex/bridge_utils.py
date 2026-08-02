"""Utilities shared by all rex bridge scripts."""

import os
import sys
import traceback as _traceback
from pathlib import Path
from typing import Any


def bridge_error_response(exc: Exception) -> dict[str, Any]:
    """Return a standard bridge error dict that includes the traceback.

    Callers should ``print(json.dumps(bridge_error_response(exc)), flush=True)``
    so that the GUI backend and CLI both receive a readable error with context.

    Do not use this for credential-bearing bridges (anything that resolves a
    vault secret, token, password, or other credential-derived value) - the
    exception message and traceback it returns can embed that secret-derived
    content verbatim. Use :func:`bridge_safe_error_response` for those.
    """
    return {
        "ok": False,
        "error": str(exc),
        "traceback": _traceback.format_exc(),
    }


def bridge_safe_error_response(
    exc: BaseException,
    *,
    messages: dict[type[BaseException], str] | None = None,
    default: str = "Request failed",
) -> dict[str, Any]:
    """Return a categorized bridge error response with no secret-derived content.

    Unlike :func:`bridge_error_response`, this never includes ``str(exc)``,
    ``repr(exc)``, a traceback, or any value derived from the exception's
    arguments - only a fixed, pre-written message selected by walking
    ``type(exc).__mro__`` against *messages* (falling back to *default* for
    anything not explicitly categorized).

    Use this for every credential-bearing bridge (vault, setup, SMS, Home
    Assistant mutation, email, calendar, and similar). Exceptions raised
    along those paths can carry provider responses, tokens, account
    identifiers, submitted request values, or filesystem paths in their
    message text; none of that may reach the renderer or CLI output.
    """
    lookup = messages or {}
    for exc_type in type(exc).__mro__:
        if exc_type in lookup:
            return {"ok": False, "error": lookup[exc_type]}
    return {"ok": False, "error": default}


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
            raise RuntimeError("Could not locate repository root (pyproject.toml not found)")
        current = parent
