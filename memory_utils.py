"""Backwards compatibility wrapper for the new :mod:`rex.memory` module."""

from rex.memory import (
    append_history_entry,
    extract_voice_reference,
    load_all_profiles,
    load_memory_profile,
    load_recent_history,
    load_users_map,
    resolve_user_key,
)

__all__ = [
    "append_history_entry",
    "extract_voice_reference",
    "load_all_profiles",
    "load_memory_profile",
    "load_recent_history",
    "load_users_map",
    "resolve_user_key",
]
