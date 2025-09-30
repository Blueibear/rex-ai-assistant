"""Persistence helpers backed by TinyDB with in-memory fallbacks."""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional

from assistant_errors import ConfigurationError
from config import AppConfig, load_config

try:  # pragma: no cover - optional dependency
    from tinydb import TinyDB
    from tinydb.table import Document

    _HAS_TINYDB = True
except ImportError:  # pragma: no cover - fallback path
    TinyDB = None  # type: ignore[assignment]
    Document = Any  # type: ignore[assignment]
    _HAS_TINYDB = False

try:  # Prefer the re-exported settings when rex is installed
    from rex import config as rex_config  # type: ignore
except Exception:  # pragma: no cover - rex package may be absent
    rex_config = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent
MEMORY_ROOT = REPO_ROOT / "Memory"
USERS_PATH = REPO_ROOT / "users.json"


class _MemoryDocument(dict):  # pragma: no cover - simple data holder
    def __init__(self, data: dict[str, Any], doc_id: int) -> None:
        super().__init__(data)
        self.doc_id = doc_id


class _MemoryTable:
    def __init__(self, store: dict[str, list[dict[str, Any]]], name: str) -> None:
        self._store = store
        self._name = name

    def insert(self, value: dict[str, Any]) -> None:
        records = self._store.setdefault(self._name, [])
        records.append(dict(value))

    def all(self) -> list[_MemoryDocument]:
        records = self._store.get(self._name, [])
        return [_MemoryDocument(record, idx + 1) for idx, record in enumerate(records)]

    def remove(self, *, doc_ids: Iterable[int]) -> None:
        ids = set(doc_ids)
        records = self._store.get(self._name, [])
        filtered = [record for idx, record in enumerate(records, start=1) if idx not in ids]
        self._store[self._name] = filtered


class _MemoryStorage:  # pragma: no cover - tiny interface shim
    def flush(self) -> None:
        return None


class _MemoryDB:
    def __init__(self) -> None:
        self._tables: dict[str, list[dict[str, Any]]] = {}

    def table(self, name: str) -> _MemoryTable:
        return _MemoryTable(self._tables, name)

    def close(self) -> None:
        return None

    @property
    def storage(self) -> _MemoryStorage:
        return _MemoryStorage()


_FALLBACK_DBS: dict[Path, _MemoryDB] = {}


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_db(path: Path) -> TinyDB | _MemoryDB:
    _ensure_directory(path.parent)
    if _HAS_TINYDB:
        return TinyDB(path, ensure_ascii=False, separators=",:")
    return _FALLBACK_DBS.setdefault(path, _MemoryDB())


def _active_settings() -> AppConfig:
    if rex_config is not None:
        return rex_config.settings  # type: ignore[attr-defined]
    return load_config()


def _default_max_memory() -> int:
    cfg = _active_settings()
    for attr in ("memory_max_turns", "max_memory_items"):
        value = getattr(cfg, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return 50


def load_users_map(users_path: str | Path = USERS_PATH) -> Dict[str, str]:
    try:
        data = json.loads(Path(users_path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}

    mapping: Dict[str, str] = {}
    if isinstance(data, dict):
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
            if not isinstance(profile, dict):
                continue
            name = profile.get("name")
            if isinstance(name, str) and name.strip().lower() == key:
                return candidate

    if Path(memory_root, key).exists():
        return key

    return None


def _profile_root() -> Path:
    cfg = _active_settings()
    return Path(cfg.user_profiles_dir or MEMORY_ROOT)


def load_memory_profile(user_key: str, memory_root: str | Path | None = None) -> dict:
    root = Path(memory_root) if memory_root else _profile_root()
    core_path = root / user_key / "core.json"
    with open(core_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_all_profiles(memory_root: str | Path | None = None) -> Dict[str, dict]:
    profiles: Dict[str, dict] = {}
    root = Path(memory_root) if memory_root else _profile_root()
    if not root.is_dir():
        return profiles

    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        core_path = candidate / "core.json"
        if not core_path.is_file():
            continue
        try:
            with core_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError:
            continue
        profiles[candidate.name] = data if isinstance(data, dict) else {}
    return profiles


def extract_voice_reference(profile: dict[str, Any]) -> Optional[str]:
    if not isinstance(profile, dict):
        return None
    voice_sample = profile.get("voice_sample")
    if isinstance(voice_sample, str):
        return voice_sample
    voice = profile.get("voice")
    if isinstance(voice, dict):
        candidate = voice.get("sample_path") or voice.get("sample")
        if isinstance(candidate, str):
            return candidate
    return None


def trim_history(history: Iterable[dict], *, limit: Optional[int] = None) -> List[dict]:
    max_items = limit or _default_max_memory()
    recent: Deque[dict] = deque(maxlen=max_items)
    for item in history:
        recent.append(item)
    return list(recent)


def append_history_entry(
    user_key: str,
    entry: Dict[str, str],
    *,
    memory_db: Path | None = None,
    max_turns: Optional[int] = None,
) -> None:
    limit = max_turns or _default_max_memory()
    if limit <= 0:
        raise ConfigurationError("History retention limit must be positive.")

    cfg = _active_settings()
    timestamp = entry.setdefault("timestamp", datetime.utcnow().isoformat())
    if "role" not in entry or "text" not in entry:
        raise ConfigurationError("History entries require 'role' and 'text' fields.")

    db_path = Path(memory_db) if memory_db else Path(cfg.memory_path)
    db = _load_db(db_path)
    try:
        table = db.table(user_key)
        table.insert(dict(entry))
        documents = table.all()
        documents.sort(key=lambda doc: doc.doc_id)
        if len(documents) > limit:
            overflow = len(documents) - limit
            doc_ids = [doc.doc_id for doc in documents[:overflow]]
            table.remove(doc_ids=doc_ids)
        db.storage.flush()
    finally:
        db.close()


def load_recent_history(
    user_key: str,
    *,
    limit: Optional[int] = None,
    memory_db: Path | None = None,
) -> List[Dict[str, str]]:
    cfg = _active_settings()
    db_path = Path(memory_db) if memory_db else Path(cfg.memory_path)
    if not db_path.exists() and _HAS_TINYDB:
        return []

    db = _load_db(db_path)
    try:
        table = db.table(user_key)
        documents = table.all()
        documents.sort(key=lambda doc: doc.doc_id)
        if limit is not None:
            documents = documents[-limit:]
        return [dict(doc) for doc in documents]
    finally:
        db.close()


def export_transcript(
    user_key: str,
    conversation: Iterable[Dict[str, str]],
    *,
    transcripts_dir: Path | None = None,
) -> Path:
    cfg = _active_settings()
    if not cfg.transcripts_enabled:
        raise ConfigurationError("Transcripts are disabled in configuration.")

    directory = Path(transcripts_dir) if transcripts_dir else Path(cfg.transcripts_dir)
    _ensure_directory(directory)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    export_path = directory / f"{user_key}-{timestamp}.json"
    data = list(conversation)
    with open(export_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return export_path


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
