"""Core package for the Rex voice assistant.

Importing :mod:`rex` used to configure logging immediately, which proved
problematic for applications that need to install their own handlers before
any log messages are emitted. The package now exposes the helper functions
without triggering side effects so callers can opt in to
:func:`rex.logging_utils.configure_logging` when appropriate.
"""

from __future__ import annotations

# Apply compatibility shims for third-party library changes
# This must be imported early to patch libraries before they're used
from .compat import ensure_transformers_compatibility  # noqa: F401

from .logging_utils import configure_logging

try:
    from .config import reload_settings, settings
except Exception as exc:  # pragma: no cover - defensive guard for import-time config failures
    _config_import_error = exc

    def reload_settings(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("Rex configuration failed to load.") from _config_import_error

    settings = None  # type: ignore[assignment]

# Credential management
from .credentials import (
    Credential,
    CredentialManager,
    CredentialRefreshError,
    get_credential_manager,
    set_credential_manager,
    mask_token,
)

# Tool registry
from .tool_registry import (
    ToolMeta,
    ToolRegistry,
    ToolNotFoundError,
    MissingCredentialError,
    get_tool_registry,
    set_tool_registry,
    register_tool,
)

__all__ = [
    # Configuration
    "settings",
    "reload_settings",
    "configure_logging",
    # Credential management
    "Credential",
    "CredentialManager",
    "CredentialRefreshError",
    "get_credential_manager",
    "set_credential_manager",
    "mask_token",
    # Tool registry
    "ToolMeta",
    "ToolRegistry",
    "ToolNotFoundError",
    "MissingCredentialError",
    "get_tool_registry",
    "set_tool_registry",
    "register_tool",
]
