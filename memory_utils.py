"""Helper utilities for loading Rex memory profiles and metadata."""

from __future__ import annotations

import json
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional

from assistant_errors import ConfigurationError
from config import load_config

REPO_ROOT = Path(__file__).resolve().parent
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
    memory_root: str | Path = MEMORY_ROOT,
    profiles: Optional[Dict[str, dict]] = None,
) -> Optional[str]:
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
    core_path = Path(memory_root) / user_key / "core.json"
    with open(core_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_all_profiles(memory_root: str | Path = MEMORY_ROOT) -> Dict[str, dict]:
    profiles: Dict[str, dict] = {}
    memory_root = Path(memory_root)
    if not memory_root.is_dir():
        return profiles

    for entry in os.listdir(memory_root):
        path = memory_root / entry
        if not path.is_dir():
            continue

        key = entry.lower()
        try:
            profile = load_memory_profile(entry, memory_root)
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        profiles[key] = profile

    return profiles


def extract_voice_reference(profile: dict) -> Optional[str]:
    voice_sample = profile.get("voice_sample")
    if isinstance(voice_sample, str):
        return voice_sample

    voice = profile.get("voice") if isinstance(profile, dict) else None
    if isinstance(voice, dict):
        candidate = voice.get("sample_path") or voice.get("sample")
        if isinstance(candidate, str):
            return candidate

    return None


def trim_history(history: Iterable[dict], *, limit: Optional[int] = None) -> List[dict]:
    """Return only the most recent `limit` entries from a history iterable."""
    from rex.config import settings
    max_items = limit or settings.max_memory_items
    recent: Deque[dict] = deque(maxlen=max_items)
    for item in history:
        recent.append(item)
    return list(recent)


def append_history_entry(
    user_key: str,
    entry: Dict[str, str],
    *,
    memory_root: str | Path = MEMORY_ROOT,
    max_turns: Optional[int] = None,
) -> None:
    cfg = load_config()
    limit = max_turns or cfg.memory_max_turns
    if limit <= 0:
        raise ConfigurationError("History retention limit must be positive.")

    timestamp = entry.setdefault("timestamp", datetime.utcnow().isoformat())
    if "role" not in entry or "text" not in entry:
        raise ConfigurationError("History entries require 'role' and 'text' fields.")

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
    limit: Optional[int] = None,
    memory_root: str | Path = MEMORY_ROOT,
) -> List[Dict[str, str]]:
    history_path = _history_path(user_key, Path(memory_root))
    if not history_path.is_file():
        return []

    with history_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    if limit is not None:
        lines = lines[-limit:]

    entries: List[Dict[str, str]] = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def export_transcript(
    user_key: str,
    conversation: Iterable[Dict[str, str]],
    *,
    transcripts_dir: Path | None = None,
) -> Path:
    cfg = load_config()
    if not cfg.transcripts_enabled:
        raise ConfigurationError("Transcript export is disabled in configuration.")

    transcripts_root = Path(transcripts_dir or cfg.transcripts_dir) / user_key
    _ensure_directory(transcripts_root)
    file_path = transcripts_root / f"{datetime.utcnow().date()}.txt"

    with file_path.open("a", encoding="utf-8") as handle:
        for turn in conversation:
            role = turn.get("role", "unknown")
            text = turn.get("text", "")
            timestamp = turn.get("timestamp") or datetime.utcnow().isoformat()
            handle.write(f"[{timestamp}] {role}: {text}\n")

    return file_path


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

