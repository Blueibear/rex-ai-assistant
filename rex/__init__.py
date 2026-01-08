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

from .config import reload_settings, settings
from .logging_utils import configure_logging

__all__ = ["settings", "reload_settings", "configure_logging"]
