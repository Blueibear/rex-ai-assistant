"""Backward compatibility wrapper - imports from rex.logging_utils.

New code should import directly from rex.logging_utils.

.. deprecated::
    Import from ``rex.logging_utils`` instead. This shim will be removed in a future cycle.
    References still exist in: audio_config.py, gui.py, gui_settings_tab.py, install.py
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from root-level 'logging_utils' is deprecated. "
    "Use 'from rex.logging_utils import ...' instead. "
    "This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export logging utilities from the rex package
from rex.logging_utils import (  # noqa: E402
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
