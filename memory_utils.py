"""Backward compatibility wrapper - imports from rex.memory_utils.

New code should import directly from rex.memory_utils.

.. deprecated::
    Import from ``rex.memory_utils`` instead. This shim will be removed in a future cycle.
    References still exist in: flask_proxy.py, gui.py, tests/test_memory_utils.py
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from root-level 'memory_utils' is deprecated. "
    "Use 'from rex.memory_utils import ...' instead. "
    "This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all utilities from the rex.memory_utils package
from rex.memory_utils import (  # noqa: E402
    append_history_entry,
    export_transcript,
    extract_voice_reference,
    load_all_profiles,
    load_memory_profile,
    load_recent_history,
    load_users_map,
    resolve_user_key,
    trim_history,
)

__all__ = [
    "load_users_map",
    "resolve_user_key",
    "load_memory_profile",
    "load_all_profiles",
    "extract_voice_reference",
    "trim_history",
    "append_history_entry",
    "load_recent_history",
    "export_transcript",
]
