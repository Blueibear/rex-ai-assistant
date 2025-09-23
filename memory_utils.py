"""Helper utilities for loading Rex memory profiles and metadata."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MEMORY_ROOT = os.path.join(REPO_ROOT, "Memory")
USERS_PATH = os.path.join(REPO_ROOT, "users.json")


def load_users_map(users_path: str = USERS_PATH) -> Dict[str, str]:
    """Return the email-to-user mapping defined in ``users.json``.

    Parameters
    ----------
    users_path:
        Optional override for the location of ``users.json``.

    Returns
    -------
    Dict[str, str]
        Normalised mapping of lowercase email addresses to lowercase user keys.
    """
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
    """Resolve a user identifier to a memory folder key.

    The identifier can be an email address, an existing memory folder name,
    or a profile name stored in ``core.json``.

    Parameters
    ----------
    identifier:
        Email address, folder name, or display name to resolve.
    users_map:
        Mapping of email addresses to folder names from :func:`load_users_map`.
    memory_root:
        Base directory that contains user memory folders.
    profiles:
        Optional mapping of folder keys to the parsed ``core.json`` contents.

    Returns
    -------
    Optional[str]
        Lowercase memory key if it can be resolved; otherwise ``None``.
    """
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
