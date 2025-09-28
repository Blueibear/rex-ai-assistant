"""Core package for the Rex voice assistant.

Importing :mod:`rex` no longer configures logging automatically.

Instead, callers should explicitly invoke:
    `rex.configure_logging()` when needed.
This avoids issues for applications that wish to customize logging behavior.
"""

from __future__ import annotations

from .config import reload_settings, settings
from .logging_utils import configure_logging

__all__ = ["settings", "reload_settings", "configure_logging"]
