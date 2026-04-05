"""SchedulingEngine: LLM-assisted meeting slot suggestions.

Given a duration and optional constraints (participants, earliest/latest
datetimes, timezone label), the engine:

1. Fetches existing calendar events via :class:`~rex.integrations.calendar_service.CalendarService`.
2. Asks the LLM to suggest three open time slots that avoid conflicts and
   respect the user's ``active_hours`` preference.
3. Returns at most 3 :class:`~rex.integrations.models.TimeSlot` objects.

In stub mode (no ``CalendarService`` or LLM configured), returns 3
hardcoded future slots so the rest of the system works immediately.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Protocol, cast, runtime_checkable

from rex.integrations.models import CalendarEvent, TimeSlot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class SchedulingBackend(Protocol):
    """Any object that can generate text (same shape as TriageBackend)."""

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
    ) -> str:
        """Return a generated text response."""
        ...


@runtime_checkable
class CalendarBackend(Protocol):
    """Minimal calendar interface required by the engine."""

    def get_events(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        """Return events overlapping [start, end)."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUM_SLOTS = 3


def _stub_slots(earliest: datetime, duration_minutes: int) -> list[TimeSlot]:
    """Return 3 hardcoded future slots starting from *earliest*."""
    slots: list[TimeSlot] = []
    cursor = earliest.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    for i in range(_NUM_SLOTS):
        start = cursor + timedelta(hours=i * 2)
        end = start + timedelta(minutes=duration_minutes)
        slots.append(TimeSlot(start=start, end=end, confidence=0.9 - i * 0.1))
    return slots


# ---------------------------------------------------------------------------
# SchedulingEngine
# ---------------------------------------------------------------------------


class SchedulingEngine:
    """Suggest open meeting time slots using calendar data and an LLM.

    Args:
        calendar: A :class:`CalendarBackend` (or ``None`` to skip busy-time
            lookup and use stub slots).
        backend: A :class:`SchedulingBackend` LLM.  When ``None`` the engine
            lazy-imports :class:`rex.llm.LanguageModel` on first use.
        active_hours: Tuple of ``(start_hour, end_hour)`` in 24-h format
            representing the user's preferred working window (default 9–18).
    """

    def __init__(
        self,
        calendar: CalendarBackend | None = None,
        backend: SchedulingBackend | None = None,
        active_hours: tuple[int, int] = (9, 18),
    ) -> None:
        self._calendar = calendar
        self._backend = backend
        self._active_hours = active_hours

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_slots(
        self,
        duration_minutes: int,
        participants: list[str] | None = None,
        earliest: datetime | None = None,
        latest: datetime | None = None,
        timezone_label: str = "UTC",
    ) -> list[TimeSlot]:
        """Return up to 3 suggested meeting slots.

        Args:
            duration_minutes: Required meeting length in minutes.
            participants: Optional list of participant email addresses (used
                as context for the LLM prompt; not used to fetch participant
                calendars).
            earliest: Start of the search window (defaults to now).
            latest: End of the search window (defaults to 7 days from now).
            timezone_label: Human-readable timezone name for the LLM prompt
                (e.g. ``"Europe/London"``).  Does not affect datetime
                arithmetic, which uses UTC throughout.

        Returns:
            List of up to 3 :class:`TimeSlot` objects ordered by start time.
        """
        now = datetime.now(UTC)
        earliest = earliest or now
        latest = latest or (now + timedelta(days=7))

        busy_events = self._get_busy_events(earliest, latest)

        if self._calendar is None and self._backend is None:
            # Pure stub mode — no calendar, no LLM
            return _stub_slots(earliest, duration_minutes)

        if self._backend is None:
            # Calendar available but no LLM → return stub slots
            return _stub_slots(earliest, duration_minutes)

        return self._ask_llm(
            duration_minutes=duration_minutes,
            participants=participants or [],
            earliest=earliest,
            latest=latest,
            timezone_label=timezone_label,
            busy_events=busy_events,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_busy_events(self, earliest: datetime, latest: datetime) -> list[CalendarEvent]:
        if self._calendar is None:
            return []
        try:
            return self._calendar.get_events(earliest, latest)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CalendarBackend.get_events failed: %s", exc)
            return []

    def _ask_llm(
        self,
        *,
        duration_minutes: int,
        participants: list[str],
        earliest: datetime,
        latest: datetime,
        timezone_label: str,
        busy_events: list[CalendarEvent],
    ) -> list[TimeSlot]:
        prompt = self._build_prompt(
            duration_minutes=duration_minutes,
            participants=participants,
            earliest=earliest,
            latest=latest,
            timezone_label=timezone_label,
            busy_events=busy_events,
        )
        backend = self._get_backend()
        try:
            raw = backend.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            return self._parse_response(raw, duration_minutes, earliest)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SchedulingEngine LLM call failed: %s — using stub slots", exc)
            return _stub_slots(earliest, duration_minutes)

    def _build_prompt(
        self,
        *,
        duration_minutes: int,
        participants: list[str],
        earliest: datetime,
        latest: datetime,
        timezone_label: str,
        busy_events: list[CalendarEvent],
    ) -> str:
        busy_lines = "\n".join(
            f"  - {e.title}: {e.start.strftime('%Y-%m-%d %H:%M')}–{e.end.strftime('%H:%M')} UTC"
            for e in busy_events
        )
        participant_str = ", ".join(participants) if participants else "no specific participants"
        active_start, active_end = self._active_hours

        return (
            "You are a scheduling assistant. Suggest exactly 3 meeting time slots "
            f"of {duration_minutes} minutes each.\n\n"
            f"Participants: {participant_str}\n"
            f"Search window: {earliest.strftime('%Y-%m-%d %H:%M')} UTC "
            f"to {latest.strftime('%Y-%m-%d %H:%M')} UTC\n"
            f"Timezone context: {timezone_label}\n"
            f"Preferred working hours: {active_start:02d}:00 – {active_end:02d}:00\n\n"
            "Existing commitments:\n"
            + (busy_lines if busy_lines else "  (none)\n")
            + "\n\nRespond with a JSON array of exactly 3 objects, each with:\n"
            '  "start": ISO 8601 datetime string (UTC)\n'
            '  "end": ISO 8601 datetime string (UTC)\n'
            '  "confidence": float between 0 and 1\n\n'
            "Avoid conflicts with existing commitments. "
            "Prefer slots within preferred working hours. "
            "Respond with ONLY valid JSON, no markdown fences."
        )

    def _parse_response(
        self, raw: str, duration_minutes: int, earliest: datetime
    ) -> list[TimeSlot]:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(line for line in lines if not line.startswith("```")).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("SchedulingEngine response is not valid JSON: %r", raw[:200])
            return _stub_slots(earliest, duration_minutes)

        if not isinstance(data, list):
            logger.warning("SchedulingEngine response is not a JSON array: %r", raw[:200])
            return _stub_slots(earliest, duration_minutes)

        slots: list[TimeSlot] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                start_raw = str(item.get("start", ""))
                end_raw = str(item.get("end", ""))
                confidence_raw = item.get("confidence", 0.8)
                start = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
                end = datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
                confidence = (
                    float(confidence_raw) if isinstance(confidence_raw, (int, float)) else 0.8
                )
                slots.append(TimeSlot(start=start, end=end, confidence=confidence))
            except (ValueError, TypeError) as exc:
                logger.warning("SchedulingEngine could not parse slot item: %s", exc)

        if not slots:
            return _stub_slots(earliest, duration_minutes)

        return sorted(slots, key=lambda s: s.start)[:_NUM_SLOTS]

    def _get_backend(self) -> SchedulingBackend:
        if self._backend is not None:
            return self._backend
        try:
            from rex.llm_client import LanguageModel

            # LanguageModel satisfies SchedulingBackend at runtime via duck typing
            backend = cast(SchedulingBackend, LanguageModel())
            self._backend = backend
            return backend
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "No scheduling backend provided and LanguageModel could not be loaded."
            ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["CalendarBackend", "SchedulingBackend", "SchedulingEngine"]
