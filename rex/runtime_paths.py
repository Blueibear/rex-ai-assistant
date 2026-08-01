"""Canonical writable runtime paths for AskRex.

Runtime state must never depend on the process working directory. Electron
sets ``ASKREX_RUNTIME_DIR`` to its per-user data directory. Source checkouts
fall back to the repository root, while installed CLI use falls back to the
platform user-data directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_RUNTIME_DIR_ENV = "ASKREX_RUNTIME_DIR"
_CONFIG_PATH_ENV = "ASKREX_CONFIG_PATH"
_ENV_PATH_ENV = "ASKREX_ENV_PATH"
_PROFILES_DIR_ENV = "ASKREX_PROFILES_DIR"
_DATA_DIR_ENV = "REX_DATA_DIR"
_MEMORY_DIR_ENV = "ASKREX_MEMORY_DIR"


def _expanded_path(raw: str | os.PathLike[str]) -> Path:
    return Path(raw).expanduser().resolve(strict=False)


def source_checkout_root(start: Path | None = None) -> Path | None:
    """Return the source checkout root when ``pyproject.toml`` is discoverable."""
    current = (start or Path(__file__)).resolve(strict=False)
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def _platform_user_data_root() -> Path:
    home = Path.home()
    if sys.platform == "win32":
        base = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA")
        return _expanded_path(base) / "AskRex" if base else home / "AppData" / "Roaming" / "AskRex"
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "AskRex"
    base = os.getenv("XDG_DATA_HOME")
    return (_expanded_path(base) if base else home / ".local" / "share") / "askrex"


def runtime_root() -> Path:
    """Return the canonical writable runtime root."""
    override = os.getenv(_RUNTIME_DIR_ENV)
    if override:
        return _expanded_path(override)
    checkout = source_checkout_root()
    return checkout if checkout is not None else _platform_user_data_root()


def _resolve(value: str | os.PathLike[str] | None, default: str) -> Path:
    if value is None:
        return runtime_root() / default
    path = Path(value).expanduser()
    return (
        path.resolve(strict=False)
        if path.is_absolute()
        else (runtime_root() / path).resolve(strict=False)
    )


def config_path(value: str | os.PathLike[str] | None = None) -> Path:
    if value is None and os.getenv(_CONFIG_PATH_ENV):
        value = os.environ[_CONFIG_PATH_ENV]
    return _resolve(value, "config/rex_config.json")


def env_path(value: str | os.PathLike[str] | None = None) -> Path:
    if value is None and os.getenv(_ENV_PATH_ENV):
        value = os.environ[_ENV_PATH_ENV]
    return _resolve(value, ".env")


def profiles_dir(value: str | os.PathLike[str] | None = None) -> Path:
    if value is None and os.getenv(_PROFILES_DIR_ENV):
        value = os.environ[_PROFILES_DIR_ENV]
    return _resolve(value, "profiles")


def data_dir(value: str | os.PathLike[str] | None = None) -> Path:
    if value is None and os.getenv(_DATA_DIR_ENV):
        value = os.environ[_DATA_DIR_ENV]
    return _resolve(value, "data")


def memory_dir(value: str | os.PathLike[str] | None = None) -> Path:
    if value is None and os.getenv(_MEMORY_DIR_ENV):
        value = os.environ[_MEMORY_DIR_ENV]
    return _resolve(value, "Memory")


__all__ = [
    "config_path",
    "data_dir",
    "env_path",
    "memory_dir",
    "profiles_dir",
    "runtime_root",
    "source_checkout_root",
]
