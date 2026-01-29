"""Conversational follow-up engine for Rex."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from rex.calendar_service import CalendarService
from rex.cue_store import CueStore, get_cue_store


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class FollowupConfig:
    enabled: bool = False
    max_per_session: int = 2
    lookback_hours: int = 72
    expire_hours: int = 168


class FollowupEngine:
    """Manage follow-up cues and prompt injection."""

    def __init__(
        self,
        *,
        cue_store: CueStore | None = None,
        calendar_service: CalendarService | None = None,
        config: FollowupConfig | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._cue_store = cue_store or get_cue_store()
        self._calendar_service = calendar_service
        self._now_fn = now_fn or _utc_now
        config = config or FollowupConfig()
        self._enabled = bool(config.enabled)
        self._max_per_session = _clamp(int(config.max_per_session), 1, 5)
        self._lookback_hours = _clamp(int(config.lookback_hours), 1, 168)
        self._expire_hours = _clamp(int(config.expire_hours), 1, 720)
        self._session_asked: set[str] = set()
        self._session_count = 0

    @classmethod
    def from_settings(
        cls,
        settings: object,
        *,
        cue_store: CueStore | None = None,
        calendar_service: CalendarService | None = None,
    ) -> "FollowupEngine":
        enabled = bool(getattr(settings, "followups_enabled", False))
        max_per_session = int(getattr(settings, "followups_max_per_session", 2))
        lookback_hours = int(getattr(settings, "followups_lookback_hours", 72))
        expire_hours = int(getattr(settings, "followups_expire_hours", 168))
        config = FollowupConfig(
            enabled=enabled,
            max_per_session=max_per_session,
            lookback_hours=lookback_hours,
            expire_hours=expire_hours,
        )
        return cls(
            cue_store=cue_store,
            calendar_service=calendar_service,
            config=config,
        )

    def _generate_calendar_cues(self, now: datetime) -> None:
        if self._calendar_service is None:
            return
        if hasattr(self._calendar_service, "generate_followup_cues"):
            self._calendar_service.generate_followup_cues(  # type: ignore[attr-defined]
                self._cue_store,
                lookback_hours=self._lookback_hours,
            )
            return
        start = now - timedelta(hours=self._lookback_hours)
        events = self._calendar_service.get_events(start, now)
        for event in events:
            if event.end_time > now:
                continue
            prompt = f"How did '{event.title}' go?"
            self._cue_store.add_cue(
                prompt=prompt,
                source="calendar",
                source_id=event.event_id,
                due_at=event.end_time,
                metadata={"event_title": event.title},
            )

    def collect_followups(self) -> list[str]:
        if not self._enabled:
            return []
        if self._session_count >= self._max_per_session:
            return []
        now = self._now_fn()
        self._cue_store.prune_expired(expire_hours=self._expire_hours, now=now)
        self._generate_calendar_cues(now)
        remaining = self._max_per_session - self._session_count
        cues = self._cue_store.get_due_cues(now, limit=remaining)
        prompts: list[str] = []
        for cue in cues:
            if cue.cue_id in self._session_asked:
                continue
            self._cue_store.mark_asked(cue.cue_id, at=now)
            self._session_asked.add(cue.cue_id)
            self._session_count += 1
            prompts.append(cue.prompt)
            if len(prompts) >= remaining:
                break
        return prompts

    def format_followups(self) -> str | None:
        prompts = self.collect_followups()
        if not prompts:
            return None
        lines = ["[Follow-up cues]"]
        lines.extend(f"- {prompt}" for prompt in prompts)
        lines.append("[/Follow-up cues]")
        return "\n".join(lines)


__all__ = ["FollowupEngine", "FollowupConfig"]
