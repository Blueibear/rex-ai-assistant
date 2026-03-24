"""Follow-up engine for Rex AI Assistant.

Integrates cues into conversations to make Rex feel more personal by
following up on calendar events, reminders, and other activities.

The follow-up engine:
- Injects cue prompts into conversations
- Tracks which cues have been asked per session
- Rate-limits cue injection (default 1 per session)
- Optionally generates cues from calendar events

Usage:
    from rex.followup_engine import get_followup_engine

    engine = get_followup_engine()
    engine.start_session(user_id="default")

    prompt = engine.get_followup_prompt(user_id="default")
    if prompt:
        # include prompt naturally early in conversation
        ...

    # after delivering the prompt
    engine.mark_current_cue_asked(user_id="default")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware and normalized to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def load_config() -> dict[str, Any]:
    """Load Rex configuration.

    This function is a thin wrapper around rex.config_manager.load_config
    that allows tests to monkeypatch it easily.

    Returns:
        The full Rex configuration dict.
    """
    try:
        from rex.config_manager import load_config as _load_config_impl

        return _load_config_impl()
    except Exception:
        return {}


def _get_config() -> dict[str, Any]:
    """Load follow-up configuration with safe defaults."""
    try:
        cfg = load_config()
        followups = cfg.get("conversation", {}).get("followups", {})
        if isinstance(followups, dict):
            return followups
        return {}
    except Exception:
        return {}


@dataclass(frozen=True)
class FollowupConfig:
    """Compatibility config object (can also be constructed from rex_config.json)."""

    enabled: bool = True
    max_per_session: int = 1
    lookback_hours: int = 72
    expire_hours: int = 168


class FollowupEngine:
    """Engine for managing follow-up cue injection into conversations.

    Supports two usage styles:
    1) Modern session-based flow:
        start_session() -> get_followup_prompt() -> mark_current_cue_asked()
    2) Legacy flow:
        collect_followups() / format_followups()

    The engine holds in-memory session state per user_id.
    """

    def __init__(
        self,
        *,
        cue_store: Any = None,
        calendar_service: Any = None,
        config: FollowupConfig | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        # Lazy imports to avoid import cycles at module import time.
        if cue_store is None:
            from rex.cue_store import get_cue_store

            cue_store = get_cue_store()

        self._cue_store = cue_store
        self._calendar_service = calendar_service
        self._now_fn = now_fn or _utc_now

        # Session state
        self._session_cues_asked: dict[str, int] = {}
        self._current_cue_id: dict[str, str | None] = {}
        self._session_asked_ids: dict[str, set[str]] = {}

        # Optional fixed config override (otherwise config is read dynamically)
        self._fixed_config = config

    @classmethod
    def from_settings(
        cls,
        settings: object,
        *,
        cue_store: Any = None,
        calendar_service: Any = None,
    ) -> FollowupEngine:
        """Compatibility constructor for older settings objects."""
        enabled = bool(getattr(settings, "followups_enabled", True))
        max_per_session = int(getattr(settings, "followups_max_per_session", 1))
        lookback_hours = int(getattr(settings, "followups_lookback_hours", 72))
        expire_hours = int(getattr(settings, "followups_expire_hours", 168))

        cfg = FollowupConfig(
            enabled=enabled,
            max_per_session=max_per_session,
            lookback_hours=lookback_hours,
            expire_hours=expire_hours,
        )
        return cls(cue_store=cue_store, calendar_service=calendar_service, config=cfg)

    def _effective_config(self) -> FollowupConfig:
        if self._fixed_config is not None:
            cfg = self._fixed_config
        else:
            raw = _get_config()
            cfg = FollowupConfig(
                enabled=bool(raw.get("enabled", True)),
                max_per_session=int(raw.get("max_per_session", 1)),
                lookback_hours=int(raw.get("lookback_hours", 72)),
                expire_hours=int(raw.get("expire_hours", 168)),
            )

        return FollowupConfig(
            enabled=bool(cfg.enabled),
            max_per_session=_clamp(int(cfg.max_per_session), 1, 5),
            lookback_hours=_clamp(int(cfg.lookback_hours), 1, 168),
            expire_hours=_clamp(int(cfg.expire_hours), 1, 720),
        )

    def start_session(self, user_id: str) -> None:
        """Start a new conversation session for a user."""
        self._session_cues_asked[user_id] = 0
        self._current_cue_id[user_id] = None
        self._session_asked_ids[user_id] = set()

        cfg = self._effective_config()
        if cfg.enabled:
            self._generate_calendar_cues(user_id=user_id, config=cfg)

    def _get_calendar_service(self) -> Any:
        if self._calendar_service is not None:
            return self._calendar_service
        try:
            from rex.calendar_service import get_calendar_service

            return get_calendar_service()
        except Exception:
            return None

    def _generate_calendar_cues(self, *, user_id: str, config: FollowupConfig) -> None:
        """Generate follow-up cues from recent calendar events.

        Supports:
        - calendar.generate_followup_cues(user_id=..., lookback_hours=..., expire_hours=...)
        - fallback: calendar.get_events(start, end)
        """
        calendar = self._get_calendar_service()
        if calendar is None:
            return

        try:
            if hasattr(calendar, "generate_followup_cues"):
                calendar.generate_followup_cues(
                    user_id=user_id,
                    lookback_hours=config.lookback_hours,
                    expire_hours=config.expire_hours,
                )
                return
        except Exception as exc:
            logger.debug("Could not generate calendar cues via generate_followup_cues: %s", exc)

        # Fallback path: build cues from events list
        try:
            now = self._now_fn()
            start = now - timedelta(hours=config.lookback_hours)
            if not hasattr(calendar, "get_events"):
                return

            events = calendar.get_events(start, now)
            for event in events:
                end_time = getattr(event, "end_time", None)
                title = getattr(event, "title", None)
                event_id = getattr(event, "event_id", None)

                if (
                    not isinstance(end_time, datetime)
                    or not isinstance(title, str)
                    or not isinstance(event_id, str)
                ):
                    continue

                end_time_utc = _ensure_utc(end_time)
                if end_time_utc > _ensure_utc(now):
                    continue

                prompt = f"How did your {title} go?"
                expires_at = end_time_utc + timedelta(hours=config.expire_hours)

                # CueStore API: add_cue(user_id, source_type, source_id, title, prompt, ...)
                self._cue_store.add_cue(
                    user_id=user_id,
                    source_type="calendar",
                    source_id=event_id,
                    title=title,
                    prompt=prompt,
                    eligible_after=end_time_utc,
                    expires_at=expires_at,
                    metadata={"event_title": title},
                )
        except Exception as exc:
            logger.debug("Could not generate calendar cues via fallback path: %s", exc)

    def get_followup_prompt(
        self,
        user_id: str,
        *,
        now: datetime | None = None,
    ) -> str | None:
        """Get a follow-up prompt for the user if available.

        The cue is NOT marked as asked until mark_current_cue_asked() is called.
        """
        cfg = self._effective_config()
        if not cfg.enabled:
            return None

        asked_count = self._session_cues_asked.get(user_id, 0)
        if asked_count >= cfg.max_per_session:
            return None

        try:
            check_time = _ensure_utc(now or self._now_fn())
            pending = self._cue_store.list_pending_cues(
                user_id=user_id,
                now=check_time,
                limit=1,
                window_hours=cfg.lookback_hours,
            )
            if not pending:
                return None

            cue = pending[0]
            self._current_cue_id[user_id] = cue.cue_id
            return cue.prompt  # type: ignore[no-any-return]
        except Exception as exc:
            logger.warning("Failed to get followup prompt: %s", exc)
            return None

    def get_followup_context(
        self,
        user_id: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Return follow-up context including prompt and metadata."""
        cfg = self._effective_config()
        if not cfg.enabled:
            return None

        asked_count = self._session_cues_asked.get(user_id, 0)
        if asked_count >= cfg.max_per_session:
            return None

        try:
            check_time = _ensure_utc(now or self._now_fn())
            pending = self._cue_store.list_pending_cues(
                user_id=user_id,
                now=check_time,
                limit=1,
                window_hours=cfg.lookback_hours,
            )
            if not pending:
                return None

            cue = pending[0]
            self._current_cue_id[user_id] = cue.cue_id
            return {
                "prompt": cue.prompt,
                "cue_id": cue.cue_id,
                "title": cue.title,
                "source_type": cue.source_type,
                "source_id": cue.source_id,
                "metadata": cue.metadata,
            }
        except Exception as exc:
            logger.warning("Failed to get followup context: %s", exc)
            return None

    def mark_current_cue_asked(self, user_id: str) -> bool:
        """Mark the current cue as asked for this user session."""
        cue_id = self._current_cue_id.get(user_id)
        if cue_id is None:
            return False

        try:
            if self._cue_store.mark_asked(cue_id):
                self._session_cues_asked[user_id] = self._session_cues_asked.get(user_id, 0) + 1
                self._current_cue_id[user_id] = None
                self._session_asked_ids.setdefault(user_id, set()).add(cue_id)
                return True
            return False
        except Exception as exc:
            logger.warning("Failed to mark cue as asked: %s", exc)
            return False

    def has_pending_cue(self, user_id: str) -> bool:
        """Check if there's at least one pending cue for the user."""
        cfg = self._effective_config()
        if not cfg.enabled:
            return False

        try:
            pending = self._cue_store.list_pending_cues(
                user_id=user_id,
                limit=1,
                window_hours=cfg.lookback_hours,
            )
            return len(pending) > 0
        except Exception:
            return False

    def get_session_cues_asked(self, user_id: str) -> int:
        """Get the number of cues asked in the current session."""
        return self._session_cues_asked.get(user_id, 0)

    def inject_followup_into_prompt(
        self,
        user_id: str,
        base_prompt: str,
        *,
        now: datetime | None = None,
    ) -> str:
        """Inject a follow-up cue into a prompt."""
        followup = self.get_followup_prompt(user_id, now=now)
        if followup is None:
            return base_prompt

        instruction = (
            "\n\nEarly in this conversation, naturally ask the user: "
            f'"{followup}" '
            "Make it feel like friendly small talk, not a checklist."
        )
        return base_prompt + instruction

    # Legacy compatibility API

    def collect_followups(self, user_id: str = "default") -> list[str]:
        """Collect follow-up prompts and mark them asked immediately.

        This matches the older behavior where collecting also consumes cues.
        """
        cfg = self._effective_config()
        if not cfg.enabled:
            return []

        asked_count = self._session_cues_asked.get(user_id, 0)
        if asked_count >= cfg.max_per_session:
            return []

        remaining = cfg.max_per_session - asked_count
        now = _ensure_utc(self._now_fn())

        prompts: list[str] = []
        asked_ids = self._session_asked_ids.setdefault(user_id, set())

        try:
            pending = self._cue_store.list_pending_cues(
                user_id=user_id,
                now=now,
                limit=max(1, remaining),
                window_hours=cfg.lookback_hours,
            )
            for cue in pending:
                if cue.cue_id in asked_ids:
                    continue
                if self._cue_store.mark_asked(cue.cue_id):
                    asked_ids.add(cue.cue_id)
                    self._session_cues_asked[user_id] = self._session_cues_asked.get(user_id, 0) + 1
                    prompts.append(cue.prompt)
                if len(prompts) >= remaining:
                    break
        except Exception as exc:
            logger.warning("Failed to collect followups: %s", exc)

        return prompts

    def format_followups(self, user_id: str = "default") -> str | None:
        """Return formatted follow-up block or None."""
        prompts = self.collect_followups(user_id=user_id)
        if not prompts:
            return None
        lines = ["[Follow-up cues]"]
        lines.extend(f"- {prompt}" for prompt in prompts)
        lines.append("[/Follow-up cues]")
        return "\n".join(lines)


# Global instance
_followup_engine: FollowupEngine | None = None


def get_followup_engine() -> FollowupEngine:
    """Get the global follow-up engine instance."""
    global _followup_engine
    if _followup_engine is None:
        _followup_engine = FollowupEngine()
    return _followup_engine


def set_followup_engine(engine: FollowupEngine | None) -> None:
    """Set the global follow-up engine instance (for testing)."""
    global _followup_engine
    _followup_engine = engine


__all__ = [
    "FollowupConfig",
    "FollowupEngine",
    "get_followup_engine",
    "set_followup_engine",
    "load_config",
]
