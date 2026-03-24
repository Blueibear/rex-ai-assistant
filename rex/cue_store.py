"""Cue storage and management for Rex AI Assistant.

Cues are lightweight records that Rex can bring up in natural conversation,
such as follow-up questions after calendar events or reminders.

Cues are:
- Per user/profile
- Persisted (survive restart)
- Consumable (asked once, then marked handled)
- Rate-limited (default max 1 cue per conversation session)
- Time-windowed (default only consider cues within last N hours/days)

Usage:
    from rex.cue_store import get_cue_store, Cue

    store = get_cue_store()
    cue = store.add_cue(
        user_id="default",
        source_type="calendar",
        source_id="event-123",
        title="Doctor appointment",
        prompt="How did your doctor appointment go?",
    )
    pending = store.list_pending_cues(user_id="default")
    store.mark_asked(cue.cue_id)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default storage directory
_DATA_DIR = Path("data/cues")


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware and normalized to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _ensure_data_dir() -> Path:
    """Ensure the data directory exists."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR


CueSourceType = Literal["calendar", "reminder", "manual"]
CueStatus = Literal["pending", "asked", "dismissed"]


class Cue(BaseModel):
    """A follow-up cue that Rex can bring up in conversation.

    Attributes:
        cue_id: Stable unique identifier for this cue.
        user_id: User/profile ID this cue belongs to.
        source_type: Type of source that generated this cue.
        source_id: ID of the source (event ID, reminder ID, etc.).
        title: Short label for the cue.
        prompt: What Rex might ask the user.
        created_at: When the cue was created (UTC).
        eligible_after: When the cue becomes eligible to be asked (UTC).
        expires_at: When the cue expires and should no longer be asked (UTC).
        status: Current status of the cue.
        asked_at: When the cue was asked (if applicable).
        dismissed_at: When the cue was dismissed (if applicable).
        metadata: Additional metadata for the cue.
    """

    cue_id: str = Field(
        default_factory=lambda: f"cue_{uuid.uuid4().hex[:12]}",
        description="Stable unique identifier for this cue",
    )
    user_id: str = Field(
        default="default",
        description="User/profile ID this cue belongs to",
    )
    source_type: CueSourceType = Field(
        ...,
        description="Type of source that generated this cue",
    )
    source_id: str = Field(
        ...,
        description="ID of the source (event ID, reminder ID, etc.)",
    )
    title: str = Field(
        ...,
        description="Short label for the cue",
    )
    prompt: str = Field(
        ...,
        description="What Rex might ask the user",
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="When the cue was created (UTC)",
    )
    eligible_after: datetime = Field(
        default_factory=_utc_now,
        description="When the cue becomes eligible to be asked (UTC)",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="When the cue expires (UTC), None for no expiration",
    )
    status: CueStatus = Field(
        default="pending",
        description="Current status of the cue",
    )
    asked_at: datetime | None = Field(
        default=None,
        description="When the cue was asked (if applicable)",
    )
    dismissed_at: datetime | None = Field(
        default=None,
        description="When the cue was dismissed (if applicable)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the cue",
    )

    def is_expired(self, now: datetime | None = None) -> bool:
        """Check if this cue has expired."""
        if self.expires_at is None:
            return False
        check_time = _ensure_utc(now or _utc_now())
        return check_time > _ensure_utc(self.expires_at)

    def is_eligible(self, now: datetime | None = None) -> bool:
        """Check if this cue is eligible to be asked."""
        check_time = _ensure_utc(now or _utc_now())
        return check_time >= _ensure_utc(self.eligible_after)

    def is_pending_and_eligible(self, now: datetime | None = None) -> bool:
        """Check if this cue is pending and eligible to be asked."""
        check_time = _ensure_utc(now or _utc_now())
        return (
            self.status == "pending"
            and self.is_eligible(check_time)
            and not self.is_expired(check_time)
        )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "cue_id": "cue_abc123def456",
                    "user_id": "default",
                    "source_type": "calendar",
                    "source_id": "event-001",
                    "title": "Doctor appointment",
                    "prompt": "How did your doctor appointment go?",
                    "created_at": "2024-01-15T10:30:00Z",
                    "eligible_after": "2024-01-15T14:00:00Z",
                    "expires_at": "2024-01-18T14:00:00Z",
                    "status": "pending",
                }
            ]
        }
    }


class CueStore:
    """Storage and management for follow-up cues.

    Provides persistence for cues with support for:
    - Adding new cues
    - Listing pending cues with time window filtering
    - Marking cues as asked or dismissed
    - Pruning expired cues
    - Duplicate prevention by source_id
    """

    def __init__(
        self,
        storage_path: Path | str | None = None,
    ) -> None:
        """Initialize the cue store.

        Args:
            storage_path: Path to the persistence file. Defaults to
                data/cues/cues.json
        """
        if storage_path is None:
            _ensure_data_dir()
            storage_path = _DATA_DIR / "cues.json"

        self.storage_path = Path(storage_path)
        self._cues: dict[str, Cue] = {}
        self._load()

    def _load(self) -> None:
        """Load cues from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, encoding="utf-8") as f:
                    data = json.load(f)
                    for cue_data in data.get("cues", []):
                        cue = Cue.model_validate(cue_data)
                        self._cues[cue.cue_id] = cue
                    logger.debug("Loaded %s cues from disk", len(self._cues))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load cues: %s", exc)
                self._cues = {}

    def _save(self) -> None:
        """Save cues to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cues_data = [cue.model_dump(mode="json") for cue in self._cues.values()]
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({"cues": cues_data}, f, indent=2, default=str)
        except OSError as exc:
            logger.error("Failed to save cues: %s", exc)

    def add_cue(
        self,
        user_id: str,
        source_type: CueSourceType,
        source_id: str,
        title: str,
        prompt: str,
        *,
        eligible_after: datetime | None = None,
        expires_at: datetime | None = None,
        expires_in: timedelta | None = None,
        metadata: dict[str, Any] | None = None,
        cue_id: str | None = None,
    ) -> Cue:
        """Add a new cue to the store.

        Returns:
            The created Cue.

        Note:
            Duplicate prevention: if an existing pending cue for the same
            (user_id, source_type, source_id) exists, it is returned.
        """
        for existing in self._cues.values():
            if (
                existing.user_id == user_id
                and existing.source_type == source_type
                and existing.source_id == source_id
                and existing.status == "pending"
            ):
                logger.debug("Cue already exists for source %s:%s", source_type, source_id)
                return existing

        now = _utc_now()

        effective_eligible = _ensure_utc(eligible_after) if eligible_after is not None else now

        effective_expires: datetime | None = None
        if expires_at is not None:
            effective_expires = _ensure_utc(expires_at)
        elif expires_in is not None:
            effective_expires = now + expires_in

        cue = Cue(
            cue_id=cue_id or f"cue_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            source_type=source_type,
            source_id=source_id,
            title=title,
            prompt=prompt,
            created_at=now,
            eligible_after=effective_eligible,
            expires_at=effective_expires,
            status="pending",
            metadata=metadata or {},
        )

        self._cues[cue.cue_id] = cue
        self._save()
        logger.debug("Added cue %s for %s:%s", cue.cue_id, source_type, source_id)
        return cue

    def get_cue(self, cue_id: str) -> Cue | None:
        """Get a specific cue by ID."""
        return self._cues.get(cue_id)

    def list_pending_cues(
        self,
        user_id: str,
        now: datetime | None = None,
        limit: int = 10,
        window_hours: int | None = None,
    ) -> list[Cue]:
        """List pending cues for a user."""
        check_time = _ensure_utc(now or _utc_now())
        window_start: datetime | None = None
        if window_hours is not None:
            window_start = check_time - timedelta(hours=window_hours)

        pending: list[Cue] = []
        for cue in self._cues.values():
            if cue.user_id != user_id:
                continue
            if not cue.is_pending_and_eligible(check_time):
                continue
            if window_start is not None and _ensure_utc(cue.created_at) < window_start:
                continue
            pending.append(cue)

        pending.sort(key=lambda c: _ensure_utc(c.created_at))
        return pending[:limit]

    def list_all_cues(
        self,
        user_id: str | None = None,
        status: CueStatus | None = None,
    ) -> list[Cue]:
        """List all cues with optional filtering."""
        cues = list(self._cues.values())
        if user_id is not None:
            cues = [c for c in cues if c.user_id == user_id]
        if status is not None:
            cues = [c for c in cues if c.status == status]
        cues.sort(key=lambda c: _ensure_utc(c.created_at), reverse=True)
        return cues

    def mark_asked(self, cue_id: str) -> bool:
        """Mark a cue as asked."""
        cue = self._cues.get(cue_id)
        if cue is None:
            return False

        cue.status = "asked"
        cue.asked_at = _utc_now()
        self._save()
        logger.debug("Marked cue %s as asked", cue_id)
        return True

    def dismiss(self, cue_id: str) -> bool:
        """Dismiss a cue."""
        cue = self._cues.get(cue_id)
        if cue is None:
            return False

        cue.status = "dismissed"
        cue.dismissed_at = _utc_now()
        self._save()
        logger.debug("Dismissed cue %s", cue_id)
        return True

    def prune_expired(self, now: datetime | None = None) -> int:
        """Remove expired cues from the store."""
        check_time = _ensure_utc(now or _utc_now())
        expired_ids = [cue_id for cue_id, cue in self._cues.items() if cue.is_expired(check_time)]

        for cue_id in expired_ids:
            del self._cues[cue_id]

        if expired_ids:
            self._save()
            logger.info("Pruned %s expired cues", len(expired_ids))

        return len(expired_ids)

    def delete_cue(self, cue_id: str) -> bool:
        """Delete a cue from the store."""
        if cue_id in self._cues:
            del self._cues[cue_id]
            self._save()
            logger.debug("Deleted cue %s", cue_id)
            return True
        return False

    def has_cue_for_source(
        self,
        user_id: str,
        source_type: CueSourceType,
        source_id: str,
    ) -> bool:
        """Check if a cue already exists for a given source."""
        for cue in self._cues.values():
            if (
                cue.user_id == user_id
                and cue.source_type == source_type
                and cue.source_id == source_id
            ):
                return True
        return False

    def __len__(self) -> int:
        """Return the number of cues in the store."""
        return len(self._cues)

    def stats(self) -> dict[str, Any]:
        """Return summary statistics for the cue store."""
        by_status: dict[str, int] = {"pending": 0, "asked": 0, "dismissed": 0}
        by_source: dict[str, int] = {"calendar": 0, "reminder": 0, "manual": 0}

        for cue in self._cues.values():
            by_status[cue.status] = by_status.get(cue.status, 0) + 1
            by_source[cue.source_type] = by_source.get(cue.source_type, 0) + 1

        return {
            "total": len(self._cues),
            "by_status": by_status,
            "by_source": by_source,
        }


# Global instance
_cue_store: CueStore | None = None


def get_cue_store() -> CueStore:
    """Get the global cue store instance."""
    global _cue_store
    if _cue_store is None:
        _cue_store = CueStore()
    return _cue_store


def set_cue_store(store: CueStore | None) -> None:
    """Set the global cue store instance (for testing)."""
    global _cue_store
    _cue_store = store


__all__ = [
    "Cue",
    "CueStore",
    "CueSourceType",
    "CueStatus",
    "get_cue_store",
    "set_cue_store",
]
