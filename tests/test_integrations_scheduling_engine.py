"""Unit tests for rex.integrations.scheduling_engine — stub mode."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from rex.integrations.models import CalendarEvent, TimeSlot
from rex.integrations.scheduling_engine import (
    CalendarBackend,
    SchedulingBackend,
    SchedulingEngine,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_backend(response: str) -> SchedulingBackend:
    mock = MagicMock(spec=SchedulingBackend)
    mock.generate.return_value = response
    return mock  # type: ignore[return-value]


def _make_calendar_backend(events: list[CalendarEvent]) -> CalendarBackend:
    mock = MagicMock(spec=CalendarBackend)
    mock.get_events.return_value = events
    return mock  # type: ignore[return-value]


def _future(hours: int = 1) -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=hours)


def _slot_json(offset_hours: int, duration: int = 60, confidence: float = 0.9) -> dict[str, Any]:
    start = _future(offset_hours)
    end = start + timedelta(minutes=duration)
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Stub mode (no calendar, no LLM)
# ---------------------------------------------------------------------------


class TestStubMode:
    def test_returns_three_slots(self) -> None:
        engine = SchedulingEngine()
        slots = engine.find_slots(duration_minutes=30)
        assert len(slots) == 3

    def test_slots_are_timeslot_instances(self) -> None:
        engine = SchedulingEngine()
        slots = engine.find_slots(duration_minutes=30)
        assert all(isinstance(s, TimeSlot) for s in slots)

    def test_slots_respect_duration(self) -> None:
        engine = SchedulingEngine()
        slots = engine.find_slots(duration_minutes=45)
        for s in slots:
            delta = (s.end - s.start).total_seconds() / 60
            assert delta == pytest.approx(45)

    def test_slots_are_in_the_future(self) -> None:
        now = datetime.now(timezone.utc)
        engine = SchedulingEngine()
        slots = engine.find_slots(duration_minutes=30)
        assert all(s.start > now for s in slots)

    def test_slots_have_confidence_between_0_and_1(self) -> None:
        engine = SchedulingEngine()
        slots = engine.find_slots(duration_minutes=30)
        assert all(0.0 <= s.confidence <= 1.0 for s in slots)

    def test_earliest_parameter_shifts_window(self) -> None:
        now = datetime.now(timezone.utc)
        earliest = now + timedelta(days=2)
        engine = SchedulingEngine()
        slots = engine.find_slots(duration_minutes=30, earliest=earliest)
        assert all(s.start >= earliest for s in slots)

    def test_slots_ordered_by_start(self) -> None:
        engine = SchedulingEngine()
        slots = engine.find_slots(duration_minutes=30)
        starts = [s.start for s in slots]
        assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# LLM mode
# ---------------------------------------------------------------------------


class TestLLMMode:
    def test_llm_response_parsed_into_slots(self) -> None:
        payload = [_slot_json(2), _slot_json(4), _slot_json(6)]
        backend = _make_llm_backend(json.dumps(payload))
        engine = SchedulingEngine(backend=backend)
        slots = engine.find_slots(duration_minutes=60)
        assert len(slots) == 3

    def test_slots_returned_sorted_by_start(self) -> None:
        payload = [_slot_json(6), _slot_json(2), _slot_json(4)]
        backend = _make_llm_backend(json.dumps(payload))
        engine = SchedulingEngine(backend=backend)
        slots = engine.find_slots(duration_minutes=60)
        starts = [s.start for s in slots]
        assert starts == sorted(starts)

    def test_max_three_slots_returned(self) -> None:
        payload = [_slot_json(2 + i) for i in range(5)]
        backend = _make_llm_backend(json.dumps(payload))
        engine = SchedulingEngine(backend=backend)
        slots = engine.find_slots(duration_minutes=60)
        assert len(slots) <= 3

    def test_markdown_fenced_json_parsed(self) -> None:
        payload = [_slot_json(2), _slot_json(4), _slot_json(6)]
        fenced = "```json\n" + json.dumps(payload) + "\n```"
        backend = _make_llm_backend(fenced)
        engine = SchedulingEngine(backend=backend)
        slots = engine.find_slots(duration_minutes=30)
        assert len(slots) == 3

    def test_invalid_json_falls_back_to_stub(self) -> None:
        backend = _make_llm_backend("not json")
        engine = SchedulingEngine(backend=backend)
        slots = engine.find_slots(duration_minutes=30)
        assert len(slots) == 3

    def test_non_array_json_falls_back_to_stub(self) -> None:
        backend = _make_llm_backend(json.dumps({"start": "2025-01-01"}))
        engine = SchedulingEngine(backend=backend)
        slots = engine.find_slots(duration_minutes=30)
        assert len(slots) == 3

    def test_llm_exception_falls_back_to_stub(self) -> None:
        mock = MagicMock(spec=SchedulingBackend)
        mock.generate.side_effect = RuntimeError("LLM unavailable")
        engine = SchedulingEngine(backend=mock)  # type: ignore[arg-type]
        slots = engine.find_slots(duration_minutes=30)
        assert len(slots) == 3

    def test_confidence_preserved(self) -> None:
        payload = [
            _slot_json(2, confidence=0.95),
            _slot_json(4, confidence=0.7),
            _slot_json(6, confidence=0.5),
        ]
        backend = _make_llm_backend(json.dumps(payload))
        engine = SchedulingEngine(backend=backend)
        slots = engine.find_slots(duration_minutes=60)
        confidences = {round(s.confidence, 2) for s in slots}
        assert 0.95 in confidences


# ---------------------------------------------------------------------------
# Calendar integration
# ---------------------------------------------------------------------------


class TestCalendarIntegration:
    def test_calendar_events_fetched(self) -> None:
        cal = _make_calendar_backend([])
        backend = _make_llm_backend(json.dumps([_slot_json(2), _slot_json(4), _slot_json(6)]))
        engine = SchedulingEngine(calendar=cal, backend=backend)
        engine.find_slots(duration_minutes=30)
        assert cal.get_events.called  # type: ignore[attr-defined]

    def test_calendar_error_does_not_raise(self) -> None:
        cal = MagicMock(spec=CalendarBackend)
        cal.get_events.side_effect = RuntimeError("Calendar unavailable")
        backend = _make_llm_backend(json.dumps([_slot_json(2), _slot_json(4), _slot_json(6)]))
        engine = SchedulingEngine(calendar=cal, backend=backend)  # type: ignore[arg-type]
        slots = engine.find_slots(duration_minutes=30)
        assert len(slots) == 3


# ---------------------------------------------------------------------------
# TimeSlot model
# ---------------------------------------------------------------------------


class TestTimeSlotModel:
    def test_model_dump_round_trip(self) -> None:
        now = datetime.now(timezone.utc)
        slot = TimeSlot(start=now, end=now + timedelta(hours=1), confidence=0.8)
        restored = TimeSlot(**slot.model_dump())
        assert restored == slot

    def test_default_confidence_is_1(self) -> None:
        now = datetime.now(timezone.utc)
        slot = TimeSlot(start=now, end=now + timedelta(hours=1))
        assert slot.confidence == 1.0
