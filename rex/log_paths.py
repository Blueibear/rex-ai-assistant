"""Shared runtime log path helpers.

Current session logs live under ``data/logs``.  The older ``logs`` directory is
kept as a legacy location so historical files can still be inspected without
being mistaken for the active GUI/runtime session log.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = Path("data/logs")
DEFAULT_RUNTIME_LOG_FILE = DEFAULT_LOG_DIR / "rex.log"
DEFAULT_ERROR_LOG_FILE = DEFAULT_LOG_DIR / "error.log"
LEGACY_RUNTIME_LOG_FILE = Path("logs/rex.log")


def resolve_repo_path(path: str | os.PathLike[str] | Path) -> Path:
    """Resolve *path* relative to the repository root when needed."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def active_runtime_log_path(config: object | None = None) -> Path:
    """Return the active runtime log file path for current sessions."""
    configured = os.getenv("REX_LOG_PATH") or getattr(config, "log_path", None)
    return resolve_repo_path(configured or DEFAULT_RUNTIME_LOG_FILE)


def active_error_log_path(config: object | None = None) -> Path:
    """Return the active error-only log file path for current sessions."""
    configured = os.getenv("REX_ERROR_LOG_PATH") or getattr(config, "error_log_path", None)
    return resolve_repo_path(configured or DEFAULT_ERROR_LOG_FILE)


def legacy_runtime_log_path() -> Path:
    """Return the historical pre-GUI runtime log path."""
    return resolve_repo_path(LEGACY_RUNTIME_LOG_FILE)


__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_LOG_DIR",
    "DEFAULT_RUNTIME_LOG_FILE",
    "DEFAULT_ERROR_LOG_FILE",
    "LEGACY_RUNTIME_LOG_FILE",
    "resolve_repo_path",
    "active_runtime_log_path",
    "active_error_log_path",
    "legacy_runtime_log_path",
]
