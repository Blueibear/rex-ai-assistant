"""Memory management for Rex AI Assistant.

This module provides memory utilities for user profiles and conversation history,
as well as structured memory capabilities:

- Working Memory: Short-term buffer for recent interactions and task summaries
- Long-Term Memory: Structured entries with categories, expiration, and search

Re-exports from memory_utils for backward compatibility.

Ownership model (US-303 per-user isolation):
- Every working-memory and long-term-memory operation requires an explicit
  validated ``user_id`` (see :func:`rex.identity.validate_user_id`).
- Stores are partitioned on disk per user::

      data/memory/<user_id>/working_memory.json
      data/memory/<user_id>/long_term_memory.json

- Missing, blank, malformed, or traversal-style identity fails closed
  (``TypeError``/``ValueError``); there is no silent fallback to ``default``,
  the active user, or any other profile.
- Process-level instances are cached in dictionaries keyed by validated
  ``user_id``; one user never receives an object backed by another user's
  path or in-memory entries.
- User IDs that differ only by case from an existing profile are rejected
  on every platform: Windows/macOS filesystems are case-insensitive, so
  "James" and "james" would otherwise silently alias one on-disk store.

Legacy unscoped-file migration:
- The pre-isolation shared files ``data/memory/working_memory.json`` and
  ``data/memory/long_term_memory.json`` belong only to the distinct
  ``default`` profile. They are never shared and never reassigned to a named
  user, and named users never read them.
- Migration runs only when a caller explicitly requests
  ``user_id="default"``. It is idempotent and crash-safe: the legacy file is
  copied into ``data/memory/default/`` via a temp file + atomic replace, and
  only after the default-profile copy exists is the original moved aside as
  ``<name>.json.pre-user-isolation.bak`` (never deleted). A failed or
  partial migration leaves the original untouched and is retried on the next
  explicit ``default`` access. To recover manually, restore the ``.bak``
  file to its original name and remove ``data/memory/default/<name>.json``.

Usage:
    from rex.memory import (
        get_working_memory,
        get_long_term_memory,
        add_user_preference,
        remember_context,
    )

    # Working memory (owner-scoped)
    wm = get_working_memory(user_id="james")
    wm.add_entry("User asked about weather")

    # Long-term memory (owner-scoped)
    ltm = get_long_term_memory(user_id="james")
    entry = ltm.add_entry("preferences", {"theme": "dark"})
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rex.identity import validate_user_id

# Re-export existing memory utilities for backward compatibility
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

logger = logging.getLogger(__name__)

# Default data directory for structured memory
_DATA_DIR = Path("data/memory")

# The distinct profile that owns pre-isolation legacy data. Never used as an
# implicit fallback; callers must select it explicitly.
DEFAULT_PROFILE = "default"

# Suffix appended to a legacy shared store file once its content has been
# migrated into the explicit ``default`` profile.
LEGACY_BACKUP_SUFFIX = ".pre-user-isolation.bak"

_WORKING_MEMORY_FILENAME = "working_memory.json"
_LONG_TERM_MEMORY_FILENAME = "long_term_memory.json"


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(UTC)


def _reject_case_conflicts(user_id: str) -> None:
    """Fail closed when *user_id* differs only by case from a known profile.

    Windows (NTFS) and macOS (APFS/HFS+) filesystems are case-insensitive,
    so two IDs differing only by case would silently alias one on-disk
    store — user "James" would read and rewrite user "james"'s files. The
    later spelling is rejected on every platform so behavior is consistent
    and fail-closed.

    Raises:
        ValueError: If a case-variant of *user_id* already exists in the
            process registries or as a memory directory on disk.
    """
    folded = user_id.casefold()
    known_ids: set[str] = set(_working_memories) | set(_long_term_memories)
    if _DATA_DIR.is_dir():
        known_ids.update(entry.name for entry in _DATA_DIR.iterdir() if entry.is_dir())
    for existing in known_ids:
        if existing != user_id and existing.casefold() == folded:
            raise ValueError(
                f"Invalid user_id: {user_id!r} (conflicts with an existing memory profile)"
            )


def _user_memory_dir(user_id: str) -> Path:
    """Return the per-user memory directory for a validated user ID."""
    user_id = validate_user_id(user_id)
    _reject_case_conflicts(user_id)
    return _DATA_DIR / user_id


def _migrate_legacy_default_store(filename: str) -> None:
    """Migrate a pre-isolation shared store file to the ``default`` profile.

    Called only when a caller explicitly requests ``user_id="default"``.
    Idempotent and crash-safe:

    1. If ``data/memory/<filename>`` does not exist, do nothing.
    2. If the default-profile copy does not exist yet, copy the legacy file
       via a temp file and atomic replace. A failure here propagates (the
       caller fails closed), leaves the original untouched, and is retried
       on the next explicit ``default`` access.
    3. Only once the default-profile copy exists is the original moved aside
       as ``<filename>.pre-user-isolation.bak`` — preserved, never deleted.
       If the move fails, the original stays in place and the move is
       retried later; migration itself is already complete.
    """
    legacy = _DATA_DIR / filename
    if not legacy.is_file():
        return

    target = _DATA_DIR / DEFAULT_PROFILE / filename
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        # PID-unique temp file: concurrent first-time migrations (e.g. two
        # GUI bridge processes) each copy privately, and os.replace is
        # atomic, so interleaved writes can never publish a corrupt store.
        tmp_copy = target.with_name(f"{target.name}.migrating.{os.getpid()}")
        try:
            shutil.copyfile(legacy, tmp_copy)
            os.replace(tmp_copy, target)
        finally:
            tmp_copy.unlink(missing_ok=True)
        logger.info("Migrated legacy shared %s into the 'default' profile store", filename)

    backup = _DATA_DIR / (filename + LEGACY_BACKUP_SUFFIX)
    if not backup.exists():
        try:
            legacy.rename(backup)
            logger.info("Preserved legacy %s as %s", filename, backup.name)
        except OSError as exc:
            logger.warning(
                "Could not move legacy %s aside: %s (original left in place)",
                filename,
                exc,
            )


# =============================================================================
# Working Memory
# =============================================================================


class WorkingMemory:
    """Short-term memory buffer for recent interactions and task summaries.

    Holds an ordered list of recent entries, automatically persisting to disk
    and loading on startup. Useful for maintaining immediate conversational
    context. Each instance is backed by exactly one owner's store.

    Attributes:
        max_entries: Maximum number of entries to retain (default 100).
        storage_path: Path to the persistence file.
        user_id: Validated owner of this store, or None for an explicit
            custom ``storage_path`` (unit-test instances).

    Example:
        wm = WorkingMemory(user_id="james")
        wm.add_entry("User asked about the weather in Dallas")
        wm.add_entry("Checked weather API - sunny, 72°F")

        recent = wm.get_recent(5)
        for entry in recent:
            print(entry)
    """

    def __init__(
        self,
        storage_path: Path | str | None = None,
        max_entries: int = 100,
        *,
        user_id: str | None = None,
    ) -> None:
        """Initialize working memory for one owner.

        Args:
            storage_path: Explicit persistence file (unit-test support). When
                omitted, a validated ``user_id`` is required and the store
                lives at ``data/memory/<user_id>/working_memory.json``.
            max_entries: Maximum entries to retain before oldest are removed.
            user_id: Owner of this store. Validated; fails closed when the
                default per-user path is used and no owner is given.

        Raises:
            ValueError: If neither ``storage_path`` nor a valid ``user_id``
                is provided, or if ``user_id`` is invalid.
        """
        if user_id is not None:
            user_id = validate_user_id(user_id)

        if storage_path is None:
            if user_id is None:
                raise ValueError("WorkingMemory requires a user_id (or an explicit storage_path)")
            if user_id == DEFAULT_PROFILE:
                _migrate_legacy_default_store(_WORKING_MEMORY_FILENAME)
            storage_path = _user_memory_dir(user_id) / _WORKING_MEMORY_FILENAME

        self.user_id = user_id
        self.storage_path = Path(storage_path)
        self.max_entries = max_entries
        self._entries: list[dict[str, Any]] = []
        self._load()

    def _owner_label(self) -> str:
        """Owner identifier for logging (never memory content)."""
        return self.user_id if self.user_id is not None else "<custom-path>"

    def _load(self) -> None:
        """Load entries from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, encoding="utf-8") as f:
                    data = json.load(f)
                    self._entries = data.get("entries", [])
                    logger.debug(
                        "Loaded %d working memory entries for owner '%s'",
                        len(self._entries),
                        self._owner_label(),
                    )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    "Failed to load working memory for owner '%s': %s",
                    self._owner_label(),
                    e,
                )
                self._entries = []

    def _save(self) -> None:
        """Save entries to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({"entries": self._entries}, f, indent=2, default=str)
        except OSError as e:
            logger.error(
                "Failed to save working memory for owner '%s': %s",
                self._owner_label(),
                e,
            )

    def add_entry(self, content: str) -> None:
        """Add a new entry to working memory.

        Args:
            content: The content to add (interaction summary, note, etc.)
        """
        entry = {
            "content": content,
            "timestamp": _utc_now().isoformat(),
        }
        self._entries.append(entry)

        # Trim to max entries
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]

        self._save()

    def get_recent(self, n: int = 10) -> list[str]:
        """Get the most recent entries.

        Args:
            n: Number of entries to retrieve.

        Returns:
            List of content strings, most recent last.
        """
        recent = self._entries[-n:] if n < len(self._entries) else self._entries
        return [entry["content"] for entry in recent]

    def get_recent_with_timestamps(self, n: int = 10) -> list[dict[str, Any]]:
        """Get recent entries with their timestamps.

        Args:
            n: Number of entries to retrieve.

        Returns:
            List of entry dicts with 'content' and 'timestamp' keys.
        """
        return self._entries[-n:] if n < len(self._entries) else self._entries.copy()

    def clear(self) -> None:
        """Clear all working memory entries for this owner."""
        self._entries = []
        self._save()
        logger.debug("Cleared working memory for owner '%s'", self._owner_label())

    def __len__(self) -> int:
        """Return the number of entries."""
        return len(self._entries)

    def stats(self) -> dict[str, Any]:
        """Return summary statistics for working memory."""
        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
        }


# =============================================================================
# Long-Term Memory
# =============================================================================


class MemoryEntry(BaseModel):
    """A structured long-term memory entry.

    Attributes:
        entry_id: Unique identifier for this entry.
        category: Category for organizing entries (e.g., 'preferences', 'facts').
        content: The stored data as a dictionary.
        created_at: When the entry was created.
        expires_at: When the entry should expire (None for no expiration).
        sensitive: Whether this entry contains sensitive data.
    """

    entry_id: str = Field(
        default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this entry",
    )
    category: str = Field(
        ...,
        description="Category for organizing entries",
    )
    content: dict[str, Any] = Field(
        default_factory=dict,
        description="The stored data",
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="When the entry was created",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="When the entry should expire (None for no expiration)",
    )
    sensitive: bool = Field(
        default=False,
        description="Whether this entry contains sensitive data",
    )

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        if self.expires_at is None:
            return False
        return _utc_now() > self.expires_at

    def to_safe_dict(self) -> dict[str, Any]:
        """Return a dictionary with sensitive content redacted."""
        data = self.model_dump()
        if self.sensitive:
            data["content"] = {"[SENSITIVE]": "Content hidden"}
        return data

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "entry_id": "mem_abc123def456",
                    "category": "preferences",
                    "content": {"theme": "dark", "language": "en"},
                    "created_at": "2024-01-15T10:30:00Z",
                    "expires_at": None,
                    "sensitive": False,
                }
            ]
        }
    }


class LongTermMemory:
    """Long-term structured memory with retention policies.

    Stores entries organized by category with support for:
    - Expiration dates and automatic cleanup
    - Sensitive data flagging
    - Keyword-based search across content

    Entries are persisted to disk as JSON. Each instance is backed by exactly
    one owner's store; entry IDs may repeat across owners without collision.

    Example:
        ltm = LongTermMemory(user_id="james")

        # Add a preference that expires in 30 days
        entry = ltm.add_entry(
            category="preferences",
            content={"theme": "dark"},
            expires_in=timedelta(days=30),
        )

        # Search for entries
        results = ltm.search(keyword="theme")

        # Clean up expired entries
        ltm.run_retention_policy()
    """

    def __init__(
        self,
        storage_path: Path | str | None = None,
        *,
        user_id: str | None = None,
    ) -> None:
        """Initialize long-term memory for one owner.

        Args:
            storage_path: Explicit persistence file (unit-test support). When
                omitted, a validated ``user_id`` is required and the store
                lives at ``data/memory/<user_id>/long_term_memory.json``.
            user_id: Owner of this store. Validated; fails closed when the
                default per-user path is used and no owner is given.

        Raises:
            ValueError: If neither ``storage_path`` nor a valid ``user_id``
                is provided, or if ``user_id`` is invalid.
        """
        if user_id is not None:
            user_id = validate_user_id(user_id)

        if storage_path is None:
            if user_id is None:
                raise ValueError("LongTermMemory requires a user_id (or an explicit storage_path)")
            if user_id == DEFAULT_PROFILE:
                _migrate_legacy_default_store(_LONG_TERM_MEMORY_FILENAME)
            storage_path = _user_memory_dir(user_id) / _LONG_TERM_MEMORY_FILENAME

        self.user_id = user_id
        self.storage_path = Path(storage_path)
        self._entries: dict[str, MemoryEntry] = {}
        self._load()
        self.run_retention_policy()

    def _owner_label(self) -> str:
        """Owner identifier for logging (never memory content)."""
        return self.user_id if self.user_id is not None else "<custom-path>"

    def _load(self) -> None:
        """Load entries from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, encoding="utf-8") as f:
                    data = json.load(f)
                    for entry_data in data.get("entries", []):
                        entry = MemoryEntry.model_validate(entry_data)
                        self._entries[entry.entry_id] = entry
                    logger.debug(
                        "Loaded %d long-term memory entries for owner '%s'",
                        len(self._entries),
                        self._owner_label(),
                    )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    "Failed to load long-term memory for owner '%s': %s",
                    self._owner_label(),
                    e,
                )
                self._entries = {}

    def _save(self) -> None:
        """Save entries to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            entries_data = [entry.model_dump(mode="json") for entry in self._entries.values()]
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({"entries": entries_data}, f, indent=2, default=str)
        except OSError as e:
            logger.error(
                "Failed to save long-term memory for owner '%s': %s",
                self._owner_label(),
                e,
            )

    def add_entry(
        self,
        category: str,
        content: dict[str, Any],
        expires_in: timedelta | None = None,
        sensitive: bool = False,
        entry_id: str | None = None,
    ) -> MemoryEntry:
        """Add a new entry to long-term memory.

        Args:
            category: Category for the entry (e.g., 'preferences', 'facts').
            content: The data to store as a dictionary.
            expires_in: Time until expiration (None for no expiration).
            sensitive: Whether this contains sensitive data.
            entry_id: Optional custom entry ID.

        Returns:
            The created MemoryEntry.
        """
        expires_at = None
        if expires_in is not None:
            expires_at = _utc_now() + expires_in

        entry = MemoryEntry(
            entry_id=entry_id or f"mem_{uuid.uuid4().hex[:12]}",
            category=category,
            content=content,
            expires_at=expires_at,
            sensitive=sensitive,
        )

        self._entries[entry.entry_id] = entry
        self._save()

        logger.debug(
            "Added memory entry %s in category '%s' for owner '%s'",
            entry.entry_id,
            category,
            self._owner_label(),
        )
        return entry

    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        """Get a specific entry by ID.

        Args:
            entry_id: The entry ID to look up.

        Returns:
            The MemoryEntry if found and not expired, else None.
        """
        entry = self._entries.get(entry_id)
        if entry is None or entry.is_expired():
            return None
        return entry

    def search(
        self,
        category: str | None = None,
        keyword: str | None = None,
        include_sensitive: bool = True,
        include_expired: bool = False,
    ) -> list[MemoryEntry]:
        """Search for entries by category and/or keyword.

        Args:
            category: Filter by category (exact match).
            keyword: Search for keyword in content keys and values (substring).
            include_sensitive: Whether to include sensitive entries.
            include_expired: Whether to include expired entries.

        Returns:
            List of matching MemoryEntry objects.
        """
        results = []
        keyword_lower = keyword.lower() if keyword else None

        for entry in self._entries.values():
            # Skip expired entries unless requested
            if not include_expired and entry.is_expired():
                continue

            # Skip sensitive entries if not requested
            if not include_sensitive and entry.sensitive:
                continue

            # Category filter
            if category is not None and entry.category != category:
                continue

            # Keyword filter - search in content keys and values
            if keyword_lower is not None:
                if not self._matches_keyword(entry.content, keyword_lower):
                    # Also check category
                    if keyword_lower not in entry.category.lower():
                        continue

            results.append(entry)

        # Sort by created_at descending (newest first)
        results.sort(key=lambda e: e.created_at, reverse=True)
        return results

    def _matches_keyword(self, content: dict[str, Any], keyword: str) -> bool:
        """Check if content matches keyword (recursive substring search)."""

        def search_value(value: Any) -> bool:
            if isinstance(value, str):
                return keyword in value.lower()
            elif isinstance(value, dict):
                for k, v in value.items():
                    if keyword in str(k).lower():
                        return True
                    if search_value(v):
                        return True
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if search_value(item):
                        return True
            else:
                return keyword in str(value).lower()
            return False

        return search_value(content)

    def forget(self, entry_id: str) -> bool:
        """Delete a specific entry.

        Args:
            entry_id: The entry ID to delete.

        Returns:
            True if the entry was deleted, False if not found.
        """
        if entry_id in self._entries:
            del self._entries[entry_id]
            self._save()
            logger.debug("Deleted memory entry %s for owner '%s'", entry_id, self._owner_label())
            return True
        return False

    def run_retention_policy(self) -> int:
        """Delete all expired entries.

        This is automatically called on startup and can be called manually.

        Returns:
            Number of entries deleted.
        """
        expired_ids = [entry_id for entry_id, entry in self._entries.items() if entry.is_expired()]

        for entry_id in expired_ids:
            del self._entries[entry_id]

        if expired_ids:
            self._save()
            logger.info(
                "Retention policy deleted %d expired entries for owner '%s'",
                len(expired_ids),
                self._owner_label(),
            )

        return len(expired_ids)

    def compact(self) -> int:
        """Remove expired entries and rewrite storage to reclaim space.

        Unlike run_retention_policy which only saves when entries were deleted,
        compact always rewrites the storage file to ensure it contains only
        current entries with no stale data.

        Returns:
            Number of expired entries removed.
        """
        removed = self.run_retention_policy()
        # Always rewrite storage to compact the file
        self._save()
        logger.info(
            "Memory store compacted for owner '%s': %d expired entries removed, "
            "%d active entries retained",
            self._owner_label(),
            removed,
            len(self),
        )
        return removed

    def list_categories(self) -> list[str]:
        """List all unique categories.

        Returns:
            Sorted list of category names.
        """
        categories = set()
        for entry in self._entries.values():
            if not entry.is_expired():
                categories.add(entry.category)
        return sorted(categories)

    def count_by_category(self) -> dict[str, int]:
        """Count non-expired entries per category.

        Returns:
            Dictionary of category name to count.
        """
        counts: dict[str, int] = {}
        for entry in self._entries.values():
            if not entry.is_expired():
                counts[entry.category] = counts.get(entry.category, 0) + 1
        return counts

    def __len__(self) -> int:
        """Return the number of non-expired entries."""
        return sum(1 for e in self._entries.values() if not e.is_expired())

    def stats(self) -> dict[str, Any]:
        """Return summary statistics for long-term memory."""
        return {
            "entries": len(self),
            "categories": self.count_by_category(),
        }


# =============================================================================
# Per-User Instance Registries
# =============================================================================

_working_memories: dict[str, WorkingMemory] = {}
_long_term_memories: dict[str, LongTermMemory] = {}


def get_working_memory(*, user_id: str) -> WorkingMemory:
    """Get the working memory instance owned by *user_id*.

    Args:
        user_id: Validated owner of the store. Required; missing or invalid
            identity fails closed (``TypeError``/``ValueError``).

    Returns:
        The per-user WorkingMemory instance (created lazily).
    """
    user_id = validate_user_id(user_id)
    instance = _working_memories.get(user_id)
    if instance is None:
        instance = WorkingMemory(user_id=user_id)
        _working_memories[user_id] = instance
    return instance


def set_working_memory(wm: WorkingMemory | None, *, user_id: str) -> None:
    """Set (or clear) the working memory instance for *user_id* (for testing)."""
    user_id = validate_user_id(user_id)
    if wm is None:
        _working_memories.pop(user_id, None)
    else:
        _reject_case_conflicts(user_id)
        _working_memories[user_id] = wm


def get_long_term_memory(*, user_id: str) -> LongTermMemory:
    """Get the long-term memory instance owned by *user_id*.

    Args:
        user_id: Validated owner of the store. Required; missing or invalid
            identity fails closed (``TypeError``/``ValueError``).

    Returns:
        The per-user LongTermMemory instance (created lazily).
    """
    user_id = validate_user_id(user_id)
    instance = _long_term_memories.get(user_id)
    if instance is None:
        instance = LongTermMemory(user_id=user_id)
        _long_term_memories[user_id] = instance
    return instance


def set_long_term_memory(ltm: LongTermMemory | None, *, user_id: str) -> None:
    """Set (or clear) the long-term memory instance for *user_id* (for testing)."""
    user_id = validate_user_id(user_id)
    if ltm is None:
        _long_term_memories.pop(user_id, None)
    else:
        _reject_case_conflicts(user_id)
        _long_term_memories[user_id] = ltm


# =============================================================================
# Convenience Functions
# =============================================================================


def add_user_preference(
    key: str,
    value: Any,
    expires_in: timedelta | None = None,
    sensitive: bool = False,
    *,
    user_id: str,
) -> MemoryEntry:
    """Add a user preference to *user_id*'s long-term memory.

    Args:
        key: Preference key (e.g., 'theme', 'language').
        value: Preference value.
        expires_in: Optional expiration time.
        sensitive: Whether this is sensitive data.
        user_id: Owner of the preference. Required; fails closed on
            missing or invalid identity.

    Returns:
        The created MemoryEntry.
    """
    ltm = get_long_term_memory(user_id=user_id)
    return ltm.add_entry(
        category="user_preferences",
        content={key: value},
        expires_in=expires_in,
        sensitive=sensitive,
    )


def get_user_preferences(key: str | None = None, *, user_id: str) -> list[MemoryEntry]:
    """Get *user_id*'s preferences from long-term memory.

    Args:
        key: Optional key to search for.
        user_id: Owner whose preferences to read. Required; fails closed on
            missing or invalid identity.

    Returns:
        List of matching preference entries.
    """
    ltm = get_long_term_memory(user_id=user_id)
    return ltm.search(category="user_preferences", keyword=key)


def add_fact(
    topic: str,
    content: dict[str, Any],
    expires_in: timedelta | None = None,
    *,
    user_id: str,
) -> MemoryEntry:
    """Add a fact to *user_id*'s long-term memory.

    Args:
        topic: The topic or subject of the fact.
        content: Fact data.
        expires_in: Optional expiration time.
        user_id: Owner of the fact. Required; fails closed on missing or
            invalid identity.

    Returns:
        The created MemoryEntry.
    """
    ltm = get_long_term_memory(user_id=user_id)
    return ltm.add_entry(
        category="facts",
        content={"topic": topic, **content},
        expires_in=expires_in,
    )


def remember_context(summary: str, *, user_id: str) -> None:
    """Add a context summary to *user_id*'s working memory.

    Args:
        summary: A summary of the current context or interaction.
        user_id: Owner of the context. Required; fails closed on missing or
            invalid identity.
    """
    wm = get_working_memory(user_id=user_id)
    wm.add_entry(summary)


def get_recent_context(n: int = 5, *, user_id: str) -> list[str]:
    """Get recent context from *user_id*'s working memory.

    Args:
        n: Number of entries to retrieve.
        user_id: Owner whose context to read. Required; fails closed on
            missing or invalid identity.

    Returns:
        List of recent context summaries.
    """
    wm = get_working_memory(user_id=user_id)
    return wm.get_recent(n)


# =============================================================================
# Per-User Maintenance (cleanup scheduling and metrics)
# =============================================================================


def list_memory_user_ids() -> list[str]:
    """Discover user IDs that have a per-user memory directory on disk.

    Only directory names that pass :func:`rex.identity.validate_user_id` are
    returned; anything else (including the legacy shared files and their
    backups at the data-dir root) is ignored and never treated as a user.
    """
    if not _DATA_DIR.is_dir():
        return []
    users: list[str] = []
    for entry in sorted(_DATA_DIR.iterdir()):
        if not entry.is_dir():
            continue
        try:
            validate_user_id(entry.name)
        except ValueError:
            logger.warning("Ignoring invalid memory directory name: %r", entry.name)
            continue
        users.append(entry.name)
    return users


def run_memory_cleanup() -> dict[str, int]:
    """Compact every known user's long-term store, each independently.

    Covers users discovered on disk plus any live registry instances (e.g.
    test-injected stores). A failure in one user's store is logged and never
    affects another user's store.

    Returns:
        Mapping of user_id to the number of expired entries removed. Users
        whose cleanup failed are omitted.
    """
    user_ids = sorted(set(list_memory_user_ids()) | set(_long_term_memories))
    results: dict[str, int] = {}
    for user_id in user_ids:
        try:
            removed = get_long_term_memory(user_id=user_id).compact()
        except Exception:
            logger.exception(
                "Memory cleanup failed for owner '%s'; other stores unaffected", user_id
            )
            continue
        results[user_id] = removed
    return results


def schedule_memory_cleanup(
    scheduler: Any,
    interval_seconds: int = 3600,
    job_id: str = "memory_cleanup",
) -> None:
    """Register a scheduled memory cleanup job with the scheduler.

    Registers a callback that runs :func:`run_memory_cleanup` at the given
    interval: every user's long-term store is compacted independently, one
    user's failure never touches another user's file, and invalid directory
    names are ignored.

    Args:
        scheduler: A Scheduler instance from rex.scheduler.
        interval_seconds: How often to run cleanup (default: 3600 = 1 hour).
        job_id: Unique ID for the scheduler job.
    """

    def _cleanup_callback(job: Any) -> None:  # noqa: ARG001
        results = run_memory_cleanup()
        removed = sum(results.values())
        logger.info(
            "Scheduled memory cleanup complete: %d expired entries removed "
            "across %d user store(s)",
            removed,
            len(results),
        )

    # Register callback and add the job
    scheduler.register_callback("memory_cleanup", _cleanup_callback)
    scheduler.add_job(
        job_id=job_id,
        name="Memory Cleanup",
        schedule=f"interval:{interval_seconds}",
        callback_name="memory_cleanup",
    )
    logger.info(f"Memory cleanup scheduled every {interval_seconds}s (job_id={job_id})")


def memory_store_metrics() -> dict[str, Any]:
    """Aggregate entry counts across all per-user stores (no content).

    Used by the service supervisor for health metrics. Counts only — never
    memory content, categories, or entry IDs.
    """
    user_ids = sorted(
        set(list_memory_user_ids()) | set(_working_memories) | set(_long_term_memories)
    )
    working_total = 0
    long_term_total = 0
    for user_id in user_ids:
        try:
            working_total += len(get_working_memory(user_id=user_id))
            long_term_total += len(get_long_term_memory(user_id=user_id))
        except Exception:
            logger.exception("Failed to collect memory metrics for owner '%s'", user_id)
    return {
        "user_count": len(user_ids),
        "working_memory_entries": working_total,
        "long_term_memory_entries": long_term_total,
    }


# Update __all__ to include new exports
__all__ = [
    # Legacy memory utilities
    "load_users_map",
    "resolve_user_key",
    "load_memory_profile",
    "load_all_profiles",
    "extract_voice_reference",
    "trim_history",
    "append_history_entry",
    "load_recent_history",
    "export_transcript",
    # Ownership constants
    "DEFAULT_PROFILE",
    "LEGACY_BACKUP_SUFFIX",
    # Working memory
    "WorkingMemory",
    "get_working_memory",
    "set_working_memory",
    # Long-term memory
    "MemoryEntry",
    "LongTermMemory",
    "get_long_term_memory",
    "set_long_term_memory",
    # Convenience functions
    "add_user_preference",
    "get_user_preferences",
    "add_fact",
    "remember_context",
    "get_recent_context",
    # Per-user maintenance
    "list_memory_user_ids",
    "run_memory_cleanup",
    "schedule_memory_cleanup",
    "memory_store_metrics",
]
