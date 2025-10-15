"""Backward compatibility wrapper - imports from rex.config.

New code should import directly from rex.config or use 'from rex import settings'.
"""

from __future__ import annotations

# Re-export all configuration utilities from the rex package
from rex.config import (
    AppConfig,
    ENV_MAPPING,
    ENV_PATH,
    REQUIRED_ENV_KEYS,
    cli,
    load_config,
    reload_settings,
    settings,
    show_config,
    validate_config,
)

__all__ = [
    "AppConfig",
    "ENV_MAPPING",
    "ENV_PATH",
    "REQUIRED_ENV_KEYS",
    "settings",
    "load_config",
    "reload_settings",
    "validate_config",
    "show_config",
    "cli",
]
