"""Helper utilities for loading Rex memory profiles and metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

from placeholder_voice import (
    DEFAULT_PLACEHOLDER_RELATIVE_PATH,
    ensure_placeholder_voice,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MEMORY_ROOT = os.path.join(REPO_ROOT, "Memory")
USERS_PATH = os.path.join(REPO_ROOT, "users.json")
MAX_MEMORY_FILE_BYTES = int(os.getenv("REX_MEMORY_MAX_BYTES", "131072"))


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
    if MAX_MEMORY_FILE_BYTES > 0:
        try:
            size = os.path.getsize(core_path)
        except OSError:
            size = 0
        if size > MAX_MEMORY_FILE_BYTES:
            raise ValueError(
                f"Memory profile '{core_path}' exceeds {MAX_MEMORY_FILE_BYTES} bytes; "
                "refusing to load to prevent runaway growth."
            )
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


def _looks_like_placeholder(path: str) -> bool:
    """Return ``True`` if ``path`` refers to the generated placeholder voice."""

    normalised = path.replace("\\", "/").strip()
    if not normalised:
        return False

    placeholder_normalised = DEFAULT_PLACEHOLDER_RELATIVE_PATH.replace("\\", "/")
    if normalised.endswith(placeholder_normalised):
        return True

    # Allow callers to pass just the filename or a shortened relative path.
    return normalised.split("/")[-1] == placeholder_normalised.split("/")[-1]


def _normalise_voice_path(
    raw_path: str,
    *,
    user_key: Optional[str] = None,
    memory_root: str = MEMORY_ROOT,
    repo_root: str = REPO_ROOT,
) -> Optional[str]:
    """Return an absolute path for ``raw_path`` if it exists on disk."""

    expanded = os.path.expanduser(raw_path)
    original = Path(expanded)

    if _looks_like_placeholder(raw_path):
        return ensure_placeholder_voice(
            DEFAULT_PLACEHOLDER_RELATIVE_PATH,
            repo_root=repo_root,
        )

    candidates = []
    if original.is_absolute():
        candidates.append(original)
    else:
        if user_key:
            candidates.append(Path(memory_root) / user_key / raw_path)
        candidates.append(Path(memory_root) / raw_path)
        candidates.append(Path(repo_root) / raw_path)
        candidates.append(Path(expanded))

    # Always check the original path last so existing absolute paths win.
    candidates.append(original)

    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved.exists():
            try:
                return str(resolved.resolve())
            except OSError:
                return str(resolved)

    return None


def extract_voice_reference(
    profile: dict,
    *,
    user_key: Optional[str] = None,
    memory_root: str = MEMORY_ROOT,
    repo_root: str = REPO_ROOT,
) -> Optional[str]:
    """Return the best available voice reference path from a profile."""

    voice_sample = profile.get("voice_sample")
    if isinstance(voice_sample, str) and voice_sample.strip():
        resolved = _normalise_voice_path(
            voice_sample.strip(),
            user_key=user_key,
            memory_root=memory_root,
            repo_root=repo_root,
        )
        if resolved:
            return resolved

    voice = profile.get("voice") if isinstance(profile, dict) else None
    if isinstance(voice, dict):
        candidate = voice.get("sample_path") or voice.get("sample")
        if isinstance(candidate, str) and candidate.strip():
            resolved = _normalise_voice_path(
                candidate.strip(),
                user_key=user_key,
                memory_root=memory_root,
                repo_root=repo_root,
            )
            if resolved:
                return resolved

    return None
