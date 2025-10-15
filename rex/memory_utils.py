"""Helper utilities for loading Rex memory profiles and metadata."""

from __future__ import annotations

import json
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional

from .assistant_errors import ConfigurationError
from .config import load_config, settings
from .placeholder_voice import (
    DEFAULT_PLACEHOLDER_RELATIVE_PATH,
    ensure_placeholder_voice,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MEMORY_ROOT = REPO_ROOT / "Memory"
USERS_PATH = REPO_ROOT / "users.json"
HISTORY_FILENAME = "history.jsonl"
HISTORY_META = "history_meta.json"
MAX_MEMORY_FILE_BYTES = int(os.getenv("REX_MEMORY_MAX_BYTES", "131072"))
MAX_PATH_DEPTH = 10  # Maximum directory traversal depth


def _sanitize_user_key(user_key: str) -> str:
    """Sanitize user key to prevent path traversal attacks."""
    if not user_key or not isinstance(user_key, str):
        raise ConfigurationError("User key must be a non-empty string.")
    
    # Remove any path separators and parent directory references
    sanitized = user_key.strip().replace("..", "").replace("/", "").replace("\\", "")
    
    # Only allow alphanumeric, dash, underscore
    if not all(c.isalnum() or c in "-_" for c in sanitized):
        raise ConfigurationError(
            f"Invalid user key '{user_key}': must contain only alphanumeric, dash, or underscore."
        )
    
    if not sanitized:
        raise ConfigurationError("User key cannot be empty after sanitization.")
    
    return sanitized.lower()


def _validate_path_within(path: Path, base: Path, *, max_depth: int = MAX_PATH_DEPTH) -> Path:
    """Validate that resolved path is within base directory and not too deep."""
    try:
        resolved = path.resolve()
        base_resolved = base.resolve()
        
        # Check if path is within base directory
        resolved.relative_to(base_resolved)
        
        # Check depth to prevent deep recursion attacks
        depth = len(resolved.relative_to(base_resolved).parts)
        if depth > max_depth:
            raise ConfigurationError(f"Path depth {depth} exceeds maximum {max_depth}.")
        
        return resolved
    except (ValueError, OSError) as exc:
        raise ConfigurationError(f"Invalid path: {path} (must be within {base})") from exc


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _history_path(user_key: str, memory_root: Path) -> Path:
    sanitized_key = _sanitize_user_key(user_key)
    user_dir = memory_root / sanitized_key
    _validate_path_within(user_dir, memory_root)
    return user_dir / HISTORY_FILENAME


def _metadata_path(user_key: str, memory_root: Path) -> Path:
    sanitized_key = _sanitize_user_key(user_key)
    user_dir = memory_root / sanitized_key
    _validate_path_within(user_dir, memory_root)
    return user_dir / HISTORY_META


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
    sanitized_key = _sanitize_user_key(user_key)
    memory_root_path = Path(memory_root)
    core_path = memory_root_path / sanitized_key / "core.json"
    
    # Validate path is within memory root
    core_path = _validate_path_within(core_path, memory_root_path)
    
    if MAX_MEMORY_FILE_BYTES > 0:
        try:
            size = core_path.stat().st_size
        except OSError:
            size = 0
        if size > MAX_MEMORY_FILE_BYTES:
            raise ConfigurationError(
                f"Memory profile '{core_path}' exceeds {MAX_MEMORY_FILE_BYTES} bytes; refusing to load."
            )
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

        # Validate entry name is safe
        try:
            sanitized = _sanitize_user_key(entry)
        except ConfigurationError:
            # Skip directories with invalid names
            continue
            
        try:
            profile = load_memory_profile(sanitized, memory_root)
        except (FileNotFoundError, json.JSONDecodeError, ConfigurationError):
            continue

        profiles[sanitized] = profile

    return profiles


def _looks_like_placeholder(path: str) -> bool:
    """Return True if the path points to the placeholder voice sample."""
    norm = path.replace("\\", "/").strip()
    if not norm:
        return False
    placeholder = DEFAULT_PLACEHOLDER_RELATIVE_PATH.replace("\\", "/")
    return norm.endswith(placeholder) or norm.split("/")[-1] == placeholder.split("/")[-1]


def _normalise_voice_path(
    raw_path: str,
    *,
    user_key: Optional[str] = None,
    memory_root: str | Path = MEMORY_ROOT,
    repo_root: str | Path = REPO_ROOT,
) -> Optional[str]:
    expanded = os.path.expanduser(raw_path)
    original = Path(expanded)

    if _looks_like_placeholder(raw_path):
        return ensure_placeholder_voice(DEFAULT_PLACEHOLDER_RELATIVE_PATH, repo_root=repo_root)

    candidates = []
    if original.is_absolute():
        candidates.append(original)
    else:
        if user_key:
            candidates.append(Path(memory_root) / user_key / raw_path)
        candidates.append(Path(memory_root) / raw_path)
        candidates.append(Path(repo_root) / raw_path)
        candidates.append(original)

    for candidate in candidates:
        try:
            resolved = candidate.expanduser()
            if resolved.exists():
                return str(resolved.resolve())
        except OSError:
            return str(candidate)

    return None


def extract_voice_reference(
    profile: dict,
    *,
    user_key: Optional[str] = None,
    memory_root: str | Path = MEMORY_ROOT,
    repo_root: str | Path = REPO_ROOT,
) -> Optional[str]:
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


def trim_history(history: Iterable[dict], *, limit: Optional[int] = None) -> List[dict]:
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

    sanitized_key = _sanitize_user_key(user_key)
    memory_root_path = Path(memory_root)
    user_dir = memory_root_path / sanitized_key
    _validate_path_within(user_dir, memory_root_path)
    _ensure_directory(user_dir)
    history_path = _history_path(sanitized_key, memory_root_path)
    metadata_path = _metadata_path(sanitized_key, memory_root_path)

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

    sanitized_key = _sanitize_user_key(user_key)
    transcripts_base = Path(transcripts_dir or cfg.transcripts_dir)
    transcripts_root = transcripts_base / sanitized_key
    _validate_path_within(transcripts_root, transcripts_base)
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

