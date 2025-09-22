"""Memory persistence and profile helpers for Rex."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import settings
from .logging_utils import configure_logger

LOGGER = configure_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
MEMORY_ROOT = REPO_ROOT / "Memory"
USERS_PATH = REPO_ROOT / "users.json"
HISTORY_FILENAME = "history.jsonl"
HISTORY_META = "history_meta.json"


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _history_path(user_key: str, memory_root: Path) -> Path:
    return memory_root / user_key / HISTORY_FILENAME


def _metadata_path(user_key: str, memory_root: Path) -> Path:
    return memory_root / user_key / HISTORY_META


def load_users_map(users_path: str | Path = USERS_PATH) -> Dict[str, str]:
    """Return the email-to-user mapping defined in ``users.json``."""

    path = Path(users_path)
    if not path.is_file():
        LOGGER.debug("No users map found at %s", path)
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    mapping: Dict[str, str] = {}
    for email, user in data.items():
        if isinstance(email, str) and isinstance(user, str):
            mapping[email.lower()] = user.lower()
    return mapping


def resolve_user_key(
    identifier: Optional[str],
    users_map: Dict[str, str],
    *,
    memory_root: str | Path = MEMORY_ROOT,
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

    if Path(memory_root, key).is_dir():
        return key

    return None


def load_memory_profile(user_key: str, memory_root: str | Path = MEMORY_ROOT) -> dict:
    """Load the ``core.json`` file for a specific user."""

    core_path = Path(memory_root) / user_key / "core.json"
    with core_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_all_profiles(memory_root: str | Path = MEMORY_ROOT) -> Dict[str, dict]:
    """Load all user profiles found in the memory directory."""

    profiles: Dict[str, dict] = {}
    memory_root = Path(memory_root)
    if not memory_root.is_dir():
        return profiles

    for entry in memory_root.iterdir():
        if not entry.is_dir():
            continue
        key = entry.name.lower()
        try:
            profiles[key] = load_memory_profile(entry.name, memory_root)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            LOGGER.warning("Skipping profile %s: %s", entry, exc)
    return profiles


def extract_voice_reference(profile: dict) -> Optional[str]:
    """Return the best available voice reference path from a profile."""

    voice_sample = profile.get("voice_sample") if isinstance(profile, dict) else None
    if isinstance(voice_sample, str):
        return voice_sample

    voice = profile.get("voice") if isinstance(profile, dict) else None
    if isinstance(voice, dict):
        candidate = voice.get("sample_path") or voice.get("sample")
        if isinstance(candidate, str):
            return candidate

    return None


def append_history_entry(
    user_key: str,
    entry: Dict[str, str],
    *,
    memory_root: str | Path = MEMORY_ROOT,
    max_turns: Optional[int] = None,
) -> None:
    """Append a structured conversation entry to ``history.jsonl``."""

    limit = max_turns or settings.memory_max_turns
    if limit <= 0:
        raise ValueError("History retention limit must be positive")

    timestamp = entry.setdefault("timestamp", datetime.utcnow().isoformat())
    if "role" not in entry or "text" not in entry:
        raise ValueError("History entries require 'role' and 'text' fields")

    memory_root = Path(memory_root)
    _ensure_directory(memory_root / user_key)
    history_path = _history_path(user_key, memory_root)
    metadata_path = _metadata_path(user_key, memory_root)

    existing: List[str] = []
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8") as handle:
            existing = handle.readlines()

    existing.append(json.dumps(entry, ensure_ascii=False) + "\n")
    trimmed = existing[-limit:]
    with history_path.open("w", encoding="utf-8") as handle:
        handle.writelines(trimmed)

    meta = {"total_turns": len(trimmed), "last_updated": timestamp}
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def load_recent_history(
    user_key: str,
    *,
    memory_root: str | Path = MEMORY_ROOT,
    limit: Optional[int] = None,
) -> List[dict]:
    """Load the ``limit`` most recent conversation turns for ``user_key``."""

    history_path = _history_path(user_key, Path(memory_root))
    if not history_path.is_file():
        return []

    with history_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    trimmed = lines[-(limit or settings.memory_max_turns) :]
    return [json.loads(line) for line in trimmed]


__all__ = [
    "append_history_entry",
    "extract_voice_reference",
    "load_all_profiles",
    "load_memory_profile",
    "load_recent_history",
    "load_users_map",
    "resolve_user_key",
]
