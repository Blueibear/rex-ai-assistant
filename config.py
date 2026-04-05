"""Backward compatibility wrapper - imports from rex.config.

New code should import directly from rex.config or use 'from rex import settings'.

.. deprecated::
    Import from ``rex.config`` instead. This shim will be removed in a future cycle.
    References still exist in: gui.py, tests/test_llm_client.py,
    tests/test_memory_utils.py, tests/test_us013_openai_provider.py,
    tests/test_us014_anthropic_provider.py, tests/test_us015_local_llm_provider.py,
    tests/test_us016_provider_routing.py
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from root-level 'config' is deprecated. "
    "Use 'from rex.config import ...' instead. "
    "This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all configuration utilities from the rex.config module
from rex.config import (
    ENV_MAPPING,
    ENV_PATH,
    REQUIRED_ENV_KEYS,
    AppConfig,
    build_app_config,
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
    "build_app_config",
    "settings",
    "load_config",
    "reload_settings",
    "validate_config",
    "show_config",
    "cli",
]
