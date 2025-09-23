"""Core package for the Rex voice assistant."""

from __future__ import annotations

from .config import reload_settings, settings
from .logging_utils import configure_logging

configure_logging()

__all__ = ["settings", "reload_settings", "configure_logging"]
