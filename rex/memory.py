"""Compatibility wrapper that re-exports memory utilities."""

from __future__ import annotations

from .memory_utils import (  # noqa: F401
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
