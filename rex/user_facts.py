"""Per-user key-value fact store persisted to Memory/<user>_facts.json.

Provides a minimal store/recall API used by the Assistant to inject remembered
facts into the system prompt:

    store("james", "dog_name", "Max")
    recall("james", "dog_name")  # -> "Max"

Each user's facts live in a single JSON file under the Memory/ directory so
they survive across sessions without touching the profile/voice data layout.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rex.identity import validate_user_id

logger = logging.getLogger(__name__)

# Memory/ is at the repo root (same directory as rex/).
_MEMORY_ROOT = Path(__file__).parent.parent / "Memory"


def _facts_path(user: str, memory_root: Path | None = None) -> Path:
    """Return the facts file for a validated user ID.

    User IDs are also used by the profile/session identity layer. Reusing its
    validator prevents ambiguous filename sanitization where distinct IDs such
    as ``alice/bob`` and ``alice_bob`` could map to the same facts file.
    """
    root = memory_root or _MEMORY_ROOT
    root.mkdir(parents=True, exist_ok=True)
    safe = validate_user_id(user)
    return root / f"{safe}_facts.json"


def _load(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load user facts from %s: %s", path, exc)
    return {}


def _save(path: Path, facts: dict[str, str]) -> None:
    try:
        path.write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.error("Failed to save user facts to %s: %s", path, exc)


def store(user: str, key: str, value: str, *, memory_root: Path | None = None) -> None:
    """Persist a key-value fact for *user*.

    Args:
        user: The user identifier (e.g. "james", "default").
        key: Fact key (e.g. "dog_name").
        value: Fact value (e.g. "Max").
        memory_root: Override the Memory/ directory (used in tests).
    """
    path = _facts_path(user, memory_root)
    facts = _load(path)
    facts[key] = value
    _save(path, facts)
    logger.debug("Stored fact for %s: %s = %s", user, key, value)


def recall(user: str, key: str, *, memory_root: Path | None = None) -> str | None:
    """Recall a previously stored fact for *user*.

    Args:
        user: The user identifier.
        key: Fact key to look up.
        memory_root: Override the Memory/ directory (used in tests).

    Returns:
        The stored value, or ``None`` if not found.
    """
    path = _facts_path(user, memory_root)
    return _load(path).get(key)


def recall_all(user: str, *, memory_root: Path | None = None) -> dict[str, str]:
    """Return all stored facts for *user* as a dict."""
    path = _facts_path(user, memory_root)
    return _load(path)


def format_facts_for_prompt(user: str, *, memory_root: Path | None = None) -> str | None:
    """Return a compact context string with all stored facts, or ``None`` if empty.

    The string is injected into the system prompt so the LLM can reference
    remembered facts during conversation.

    Example output::

        [Remembered facts about james: dog_name=Max; city=Austin]
    """
    facts = recall_all(user, memory_root=memory_root)
    if not facts:
        return None
    pairs = "; ".join(f"{k}={v}" for k, v in facts.items())
    return f"[Remembered facts about {user}: {pairs}]"
