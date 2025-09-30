"""Core package for the Rex voice assistant.

Importing :mod:`rex` no longer configures logging immediately,
which allows applications to install their own handlers before
any log messages are emitted.

To configure logging manually, call:
    rex.configure_logging()
"""

from __future__ import annotations

from .config import reload_settings, settings
from .logging_utils import configure_logging

__all__ = ["settings", "reload_settings", "configure_logging"]

