"""Compatibility shim for legacy imports of the configuration module."""

from __future__ import annotations

from rex.config import (  # noqa: F401
    Settings,
    reload_settings,
    settings,
    update_env_value,
)

__all__ = ["Settings", "settings", "reload_settings", "update_env_value"]
