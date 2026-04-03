"""In-memory chat history store with optional JSON file persistence.

Used by the Rex GUI chat API (rex/gui_app.py) to persist conversation
history across page reloads.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ChatMessage",
    "configure_store",
    "add_message",
    "get_history",
    "clear_history",
]


@dataclass
class ChatMessage:
    id: str
    role: str  # "user" | "assistant"
    content: str
    timestamp: float
    attachment_name: str | None = None


# Module-level state
_HISTORY: list[ChatMessage] = []
_STORE_PATH: Path | None = None


def configure_store(path: Path | None = None) -> None:
    """Set the optional file path for persistence and load existing data."""
    global _STORE_PATH
    _STORE_PATH = path
    if path and path.exists():
        _load()


def _load() -> None:
    global _HISTORY
    try:
        raw = json.loads(_STORE_PATH.read_text(encoding="utf-8"))  # type: ignore[union-attr]
        _HISTORY = [ChatMessage(**m) for m in raw]
    except Exception:
        _HISTORY = []


def _save() -> None:
    if _STORE_PATH is None:
        return
    try:
        _STORE_PATH.write_text(
            json.dumps([asdict(m) for m in _HISTORY], indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def add_message(
    role: str,
    content: str,
    attachment_name: str | None = None,
) -> ChatMessage:
    """Append a message to history and persist it."""
    msg = ChatMessage(
        id=str(uuid.uuid4()),
        role=role,
        content=content,
        timestamp=time.time(),
        attachment_name=attachment_name,
    )
    _HISTORY.append(msg)
    _save()
    return msg


def get_history() -> list[dict[str, Any]]:
    """Return all messages as plain dicts (JSON-serialisable)."""
    return [asdict(m) for m in _HISTORY]


def clear_history() -> None:
    """Remove all messages from memory and the backing file."""
    global _HISTORY
    _HISTORY = []
    _save()
