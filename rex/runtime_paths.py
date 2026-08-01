"""Canonical writable runtime paths for AskRex.

Runtime state must never depend on the process working directory. Electron
sets ``ASKREX_RUNTIME_DIR`` to its per-user data directory. Source checkouts
fall back to the repository root, while installed CLI use falls back to the
platform user-data directory.

Within ``data/``, household-shared state and private per-Rex-user state are
separated explicitly::

    data/household/...          shared configuration and service state
    data/users/<user_id>/...    private user-owned state
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
_HOUSEHOLD_DATA_DIR_ENV = "ASKREX_HOUSEHOLD_DATA_DIR"
_USERS_DATA_DIR_ENV = "ASKREX_USERS_DATA_DIR"
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
        # LOCALAPPDATA is the correct boundary for machine-local application
        # state. APPDATA is only a compatibility fallback.
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
        return _expanded_path(base) / "AskRex" if base else home / "AppData" / "Local" / "AskRex"
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
        return (runtime_root() / default).resolve(strict=False)
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


def household_data_dir(value: str | os.PathLike[str] | None = None) -> Path:
    """Return shared state, preserving the legacy ``REX_DATA_DIR`` contract."""
    if value is None and os.getenv(_HOUSEHOLD_DATA_DIR_ENV):
        value = os.environ[_HOUSEHOLD_DATA_DIR_ENV]
    if value is not None:
        return _resolve(value, "data/household")
    if os.getenv(_DATA_DIR_ENV):
        return data_dir()
    return _resolve(None, "data/household")


def users_data_dir(value: str | os.PathLike[str] | None = None) -> Path:
    """Return the parent directory for private Rex-user state."""
    if value is None and os.getenv(_USERS_DATA_DIR_ENV):
        value = os.environ[_USERS_DATA_DIR_ENV]
    if value is not None:
        return _resolve(value, "data/users")
    return (data_dir() / "users").resolve(strict=False)


def user_data_dir(user_id: str) -> Path:
    """Return private storage for a validated Rex user."""
    # Lazy import avoids a config/runtime_paths import cycle.
    from rex.identity import validate_user_id

    return users_data_dir() / validate_user_id(user_id)


def household_data_path(*parts: str | os.PathLike[str]) -> Path:
    """Resolve a path beneath household-shared storage."""
    return household_data_dir().joinpath(*map(Path, parts)).resolve(strict=False)


def user_data_path(user_id: str, *parts: str | os.PathLike[str]) -> Path:
    """Resolve a path beneath one validated user's private storage."""
    return user_data_dir(user_id).joinpath(*map(Path, parts)).resolve(strict=False)


def memory_dir(value: str | os.PathLike[str] | None = None) -> Path:
    if value is None and os.getenv(_MEMORY_DIR_ENV):
        value = os.environ[_MEMORY_DIR_ENV]
    return _resolve(value, "Memory")


__all__ = [
    "config_path",
    "data_dir",
    "env_path",
    "household_data_dir",
    "household_data_path",
    "memory_dir",
    "profiles_dir",
    "runtime_root",
    "source_checkout_root",
    "user_data_dir",
    "user_data_path",
    "users_data_dir",
]
