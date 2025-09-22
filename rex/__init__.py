"""Core package for the Rex voice assistant."""

from __future__ import annotations

import logging

# Configure a package level logger as soon as the package is imported.  This
# ensures every module gets consistent formatting without each one needing to
# call ``basicConfig`` individually.
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# Re-export the lazy settings instance so callers can simply import
# ``from rex import settings``.
from .config import settings  # noqa: E402

__all__ = ["settings"]
