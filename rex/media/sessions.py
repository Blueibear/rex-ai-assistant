"""Bounded in-memory state for user-owned active media sessions."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock

from rex.context.active import ActiveContextRef, ActiveContextStore
from rex.identity import validate_user_id

_PROVIDER_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MAX_TARGET_ID_LENGTH = 256
_MAX_MEDIA_REF_LENGTH = 512


def _validate_provider(provider: str) -> str:
    if not isinstance(provider, str) or not _PROVIDER_PATTERN.fullmatch(provider):
        raise ValueError("Active media session provider is invalid")
    return provider


def _validate_bounded_text(value: str, *, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(character.isspace() for character in value)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"Active media session {field} is invalid")
    return value


def _validate_timestamp(value: float, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Active media session {field} is invalid")
    value = float(value)
    if value < 0 or not math.isfinite(value):
        raise ValueError(f"Active media session {field} is invalid")
    return value


@dataclass(frozen=True, slots=True)
class ActiveMediaSession:
    """Minimal follow-up context for one user's current media interaction."""

    user_id: str
    target_id: str
    provider: str
    media_ref: str
    updated_at: float

    def __post_init__(self) -> None:
        validate_user_id(self.user_id)
        _validate_bounded_text(
            self.target_id,
            field="target_id",
            maximum=_MAX_TARGET_ID_LENGTH,
        )
        if (
            ":" not in self.target_id
            or self.target_id.startswith(":")
            or self.target_id.endswith(":")
        ):
            raise ValueError("Active media session target_id is invalid")
        _validate_provider(self.provider)
        _validate_bounded_text(
            self.media_ref,
            field="media_ref",
            maximum=_MAX_MEDIA_REF_LENGTH,
        )
        object.__setattr__(
            self,
            "updated_at",
            _validate_timestamp(self.updated_at, field="updated_at"),
        )


class ActiveMediaSessionStore:
    """Thread-safe, per-user active sessions with read-time TTL eviction."""

    def __init__(
        self,
        ttl_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        active_context_store: ActiveContextStore | None = None,
    ) -> None:
        ttl_seconds = _validate_timestamp(ttl_seconds, field="ttl_seconds")
        if ttl_seconds == 0:
            raise ValueError("Active media session ttl_seconds must be positive")
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._active_context_store = active_context_store
        self._sessions: dict[str, ActiveMediaSession] = {}
        self._lock = RLock()

    def set(self, session: ActiveMediaSession) -> None:
        """Set the requesting user's latest bounded session state."""
        if not isinstance(session, ActiveMediaSession):
            raise TypeError("session must be an ActiveMediaSession")
        user_id = validate_user_id(session.user_id)
        with self._lock:
            self._sessions[user_id] = session
        self._publish_active_context(session)

    def _publish_active_context(self, session: ActiveMediaSession) -> None:
        store = self._active_context_store
        if store is None:
            return
        try:
            store.put(
                ActiveContextRef(
                    domain="media",
                    key=session.target_id,
                    owner_user_id=session.user_id,
                    payload={
                        "target_id": session.target_id,
                        "provider": session.provider,
                        "media_ref": session.media_ref,
                    },
                    source_ids=(),
                    revision="media:1",
                    expires_at=session.updated_at + self._ttl_seconds,
                )
            )
        except (PermissionError, TypeError, ValueError):
            # Context retention is convenience state; it must never rewrite the
            # truthful result of the already-completed media operation.
            return

    def get(self, user_id: str, *, now: float | None = None) -> ActiveMediaSession | None:
        """Return one user's unexpired session and evict it when stale."""
        user_id = validate_user_id(user_id)
        current = self._clock() if now is None else now
        current = _validate_timestamp(current, field="now")
        stale: ActiveMediaSession | None = None
        with self._lock:
            session = self._sessions.get(user_id)
            if session is None:
                return None
            age = current - session.updated_at
            if age < 0 or age >= self._ttl_seconds:
                stale = self._sessions.pop(user_id)
            else:
                return session
        self._remove_active_context(stale)
        return None

    def _remove_active_context(self, session: ActiveMediaSession | None) -> None:
        if session is None or self._active_context_store is None:
            return
        self._active_context_store.remove(session.user_id, "media", session.target_id)

    def clear(self, user_id: str) -> None:
        """Remove only the requested user's active session."""
        user_id = validate_user_id(user_id)
        with self._lock:
            session = self._sessions.pop(user_id, None)
        self._remove_active_context(session)


__all__ = ["ActiveMediaSession", "ActiveMediaSessionStore"]
