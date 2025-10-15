"""Backward compatibility wrapper - imports from rex.logging_utils.

New code should import directly from rex.logging_utils.
"""

from __future__ import annotations

# Re-export logging utilities from the rex package
from rex.logging_utils import (
    LOG_FORMAT,
    configure_logging,
    get_logger,
    set_global_level,
)

__all__ = [
    "LOG_FORMAT",
    "configure_logging",
    "get_logger",
    "set_global_level",
]
