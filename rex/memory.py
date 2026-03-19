"""Memory management for Rex AI Assistant.

This module provides memory utilities for user profiles and conversation history,
as well as structured memory capabilities:

- Working Memory: Short-term buffer for recent interactions and task summaries
- Long-Term Memory: Structured entries with categories, expiration, and search

Re-exports from memory_utils for backward compatibility.

Usage:
    from rex.memory import (
        get_working_memory,
        get_long_term_memory,
        add_user_preference,
        remember_context,
    )

    # Working memory
    wm = get_working_memory()
    wm.add_entry("User asked about weather")

    # Long-term memory
    ltm = get_long_term_memory()
    entry = ltm.add_entry("preferences", {"theme": "dark"})
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

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


def _ensure_data_dir() -> Path:
    """Ensure the data directory exists."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


# =============================================================================
# Working Memory
# =============================================================================


class WorkingMemory:
    """Short-term memory buffer for recent interactions and task summaries.

    Holds an ordered list of recent entries, automatically persisting to disk
    and loading on startup. Useful for maintaining immediate conversational
    context.

    Attributes:
        max_entries: Maximum number of entries to retain (default 100).
        storage_path: Path to the persistence file.

    Example:
        wm = WorkingMemory()
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
    ) -> None:
        """Initialize working memory.

        Args:
            storage_path: Path to the persistence file. Defaults to
                data/memory/working_memory.json
            max_entries: Maximum entries to retain before oldest are removed.
        """
        if storage_path is None:
            _ensure_data_dir()
            storage_path = _DATA_DIR / "working_memory.json"

        self.storage_path = Path(storage_path)
        self.max_entries = max_entries
        self._entries: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load entries from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, encoding="utf-8") as f:
                    data = json.load(f)
                    self._entries = data.get("entries", [])
                    logger.debug(f"Loaded {len(self._entries)} working memory entries")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load working memory: {e}")
                self._entries = []

    def _save(self) -> None:
        """Save entries to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({"entries": self._entries}, f, indent=2, default=str)
        except OSError as e:
            logger.error(f"Failed to save working memory: {e}")

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
        """Clear all working memory entries."""
        self._entries = []
        self._save()

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

    Entries are persisted to disk as JSON.

    Example:
        ltm = LongTermMemory()

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
    ) -> None:
        """Initialize long-term memory.

        Args:
            storage_path: Path to the persistence file. Defaults to
                data/memory/long_term_memory.json
        """
        if storage_path is None:
            _ensure_data_dir()
            storage_path = _DATA_DIR / "long_term_memory.json"

        self.storage_path = Path(storage_path)
        self._entries: dict[str, MemoryEntry] = {}
        self._load()
        self.run_retention_policy()

    def _load(self) -> None:
        """Load entries from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, encoding="utf-8") as f:
                    data = json.load(f)
                    for entry_data in data.get("entries", []):
                        entry = MemoryEntry.model_validate(entry_data)
                        self._entries[entry.entry_id] = entry
                    logger.debug(f"Loaded {len(self._entries)} long-term memory entries")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load long-term memory: {e}")
                self._entries = {}

    def _save(self) -> None:
        """Save entries to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            entries_data = [entry.model_dump(mode="json") for entry in self._entries.values()]
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({"entries": entries_data}, f, indent=2, default=str)
        except OSError as e:
            logger.error(f"Failed to save long-term memory: {e}")

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

        logger.debug(f"Added memory entry: {entry.entry_id} in category '{category}'")
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
            logger.debug(f"Deleted memory entry: {entry_id}")
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
            logger.info(f"Retention policy deleted {len(expired_ids)} expired entries")

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
            f"Memory store compacted: {removed} expired entries removed, {len(self)} active entries retained"
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
# Global Instances (Singleton Pattern)
# =============================================================================

_working_memory: WorkingMemory | None = None
_long_term_memory: LongTermMemory | None = None


def get_working_memory() -> WorkingMemory:
    """Get the global working memory instance."""
    global _working_memory
    if _working_memory is None:
        _working_memory = WorkingMemory()
    return _working_memory


def set_working_memory(wm: WorkingMemory) -> None:
    """Set the global working memory instance (for testing)."""
    global _working_memory
    _working_memory = wm


def get_long_term_memory() -> LongTermMemory:
    """Get the global long-term memory instance."""
    global _long_term_memory
    if _long_term_memory is None:
        _long_term_memory = LongTermMemory()
    return _long_term_memory


def set_long_term_memory(ltm: LongTermMemory) -> None:
    """Set the global long-term memory instance (for testing)."""
    global _long_term_memory
    _long_term_memory = ltm


# =============================================================================
# Convenience Functions
# =============================================================================


def add_user_preference(
    key: str,
    value: Any,
    expires_in: timedelta | None = None,
    sensitive: bool = False,
) -> MemoryEntry:
    """Add a user preference to long-term memory.

    Args:
        key: Preference key (e.g., 'theme', 'language').
        value: Preference value.
        expires_in: Optional expiration time.
        sensitive: Whether this is sensitive data.

    Returns:
        The created MemoryEntry.
    """
    ltm = get_long_term_memory()
    return ltm.add_entry(
        category="user_preferences",
        content={key: value},
        expires_in=expires_in,
        sensitive=sensitive,
    )


def get_user_preferences(key: str | None = None) -> list[MemoryEntry]:
    """Get user preferences from long-term memory.

    Args:
        key: Optional key to search for.

    Returns:
        List of matching preference entries.
    """
    ltm = get_long_term_memory()
    return ltm.search(category="user_preferences", keyword=key)


def add_fact(
    topic: str,
    content: dict[str, Any],
    expires_in: timedelta | None = None,
) -> MemoryEntry:
    """Add a fact to long-term memory.

    Args:
        topic: The topic or subject of the fact.
        content: Fact data.
        expires_in: Optional expiration time.

    Returns:
        The created MemoryEntry.
    """
    ltm = get_long_term_memory()
    return ltm.add_entry(
        category="facts",
        content={"topic": topic, **content},
        expires_in=expires_in,
    )


def remember_context(summary: str) -> None:
    """Add a context summary to working memory.

    Args:
        summary: A summary of the current context or interaction.
    """
    wm = get_working_memory()
    wm.add_entry(summary)


def get_recent_context(n: int = 5) -> list[str]:
    """Get recent context from working memory.

    Args:
        n: Number of entries to retrieve.

    Returns:
        List of recent context summaries.
    """
    wm = get_working_memory()
    return wm.get_recent(n)


def schedule_memory_cleanup(
    scheduler: Any,
    interval_seconds: int = 3600,
    job_id: str = "memory_cleanup",
) -> None:
    """Register a scheduled memory cleanup job with the scheduler.

    Registers a callback that runs compact() on the global long-term memory
    at the given interval. Expired entries are removed and the store is
    compacted on each run.

    Args:
        scheduler: A Scheduler instance from rex.scheduler.
        interval_seconds: How often to run cleanup (default: 3600 = 1 hour).
        job_id: Unique ID for the scheduler job.
    """

    def _cleanup_callback(job: Any) -> None:  # noqa: ARG001
        ltm = get_long_term_memory()
        removed = ltm.compact()
        logger.info(f"Scheduled memory cleanup complete: {removed} entries removed")

    # Register callback and add the job
    scheduler.register_callback("memory_cleanup", _cleanup_callback)
    scheduler.add_job(
        job_id=job_id,
        name="Memory Cleanup",
        schedule=f"interval:{interval_seconds}",
        callback_name="memory_cleanup",
    )
    logger.info(f"Memory cleanup scheduled every {interval_seconds}s (job_id={job_id})")


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
    "schedule_memory_cleanup",
]
