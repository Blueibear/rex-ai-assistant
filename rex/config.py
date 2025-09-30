"""Compatibility wrapper that re-exports the root configuration module."""

from __future__ import annotations

from config import (  # noqa: F401 - re-exported symbols
    AppConfig,
    ENV_MAPPING,
    configure_cli,
    load_config,
    persist_override,
    reload_settings,
    settings,
    update_env_value,
)

Settings = AppConfig

__all__ = [
    "AppConfig",
    "Settings",
    "ENV_MAPPING",
    "configure_cli",
    "load_config",
    "persist_override",
    "reload_settings",
    "settings",
    "update_env_value",
]
