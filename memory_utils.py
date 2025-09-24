"""Helper utilities for loading Rex memory profiles and metadata."""

from __future__ import annotations

import json
import os
from collections import deque
from typing import Deque, Dict, Iterable, Optional

from rex.config import settings

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MEMORY_ROOT = os.path.join(REPO_ROOT, "Memory")
USERS_PATH = os.path.join(REPO_ROOT, "users.json")


def load_users_map(users_path: str = USERS_PATH) -> Dict[str, str]:
    """Return the email-to-user mapping defined in ``users.json``."""
    try:
        with open(users_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}

    mapping: Dict[str, str] = {}
    for email, user in data.items():
        if isinstance(email, str) and isinstance(user, str):
            mapping[email.lower()] = user.lower()
    return mapping


def resolve_user_key(
    identifier: Optional[str],
    users_map: Dict[str, str],
    *,
    memory_root: str = MEMORY_ROOT,
    profiles: Optional[Dict[str, dict]] = None,
) -> Optional[str]:
    """Resolve a user identifier to a memory folder key."""
    if not identifier:
        return None

    key = identifier.strip().lower()
    if profiles and key in profiles:
        return key

    if key in users_map.values():
        return key

    mapped = users_map.get(key)
    if mapped:
        return mapped

    if profiles:
        for candidate, profile in profiles.items():
            name = profile.get("name") if isinstance(profile, dict) else None
            if isinstance(name, str) and name.strip().lower() == key:
                return candidate

    if os.path.isdir(os.path.join(memory_root, key)):
        return key

    return None


def load_memory_profile(user_key: str, memory_root: str = MEMORY_ROOT) -> dict:
    """Load the ``core.json`` file for a specific user."""
    core_path = os.path.join(memory_root, user_key, "core.json")
    with open(core_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_all_profiles(memory_root: str = MEMORY_ROOT) -> Dict[str, dict]:
    """Load all user profiles found in the memory directory."""
    profiles: Dict[str, dict] = {}
    if not os.path.isdir(memory_root):
        return profiles

    for entry in os.listdir(memory_root):
        path = os.path.join(memory_root, entry)
        if not os.path.isdir(path):
            continue

        key = entry.lower()
        try:
            profile = load_memory_profile(entry, memory_root)
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        profiles[key] = profile

    return profiles


def extract_voice_reference(profile: dict) -> Optional[str]:
    """Return the best available voice reference path from a profile."""
    voice_sample = profile.get("voice_sample")
    if isinstance(voice_sample, str):
        return voice_sample

    voice = profile.get("voice") if isinstance(profile, dict) else None
    if isinstance(voice, dict):
        candidate = voice.get("sample_path") or voice.get("sample")
        if isinstance(candidate, str):
            return candidate

    return None


def trim_history(history: Iterable[dict], *, limit: Optional[int] = None) -> list[dict]:
    """Return only the most recent ``limit`` entries from ``history``."""

    max_items = limit or settings.max_memory_items
    recent: Deque[dict] = deque(maxlen=max_items)
    for item in history:
        recent.append(item)
    return list(recent)


__all__ = [
    "load_users_map",
    "resolve_user_key",
    "load_memory_profile",
    "load_all_profiles",
    "extract_voice_reference",
    "trim_history",
]
