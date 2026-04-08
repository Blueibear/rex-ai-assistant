"""Tests for US-036: Surface proactive suggestions to the user.

Covers:
- At most one suggestion per session
- Spoken text format
- Accept flow (yes → automation saved)
- Dismiss flow (no thanks → dismissal recorded, not re-suggested for 30 days)
- Already-dismissed pattern is skipped
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rex.suggestions.engine import SuggestionEngine, _build_spoken_text, _pattern_key
from rex.suggestions.pattern_detector import PatternEntry, detect_patterns

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pattern(
    entity_id: str = "light.kitchen_ceiling",
    service: str = "turn_on",
    start_hour: int = 7,
    frequency: int = 5,
) -> dict[str, Any]:
    """Build a minimal pattern dict matching detect_patterns() output."""
    return {
        "pattern": f"{service.replace('_', ' ')} {entity_id.split('.')[-1].replace('_', ' ')} around {start_hour:02d}:00",
        "frequency": frequency,
        "suggested_automation": f"Automate: {service} {entity_id} daily at {start_hour:02d}:00",
        "entity_id": entity_id,
        "service": service,
        "start_hour": start_hour,
    }


# ---------------------------------------------------------------------------
# _build_spoken_text
# ---------------------------------------------------------------------------


class TestBuildSpokenText:
    def test_7am_format(self) -> None:
        pattern = _make_pattern(
            entity_id="light.kitchen_ceiling",
            service="turn_on",
            start_hour=7,
        )
        text = _build_spoken_text(pattern)
        assert "turn on" in text
        assert "kitchen ceiling" in text
        assert "7am" in text
        assert "Want me to automate that?" in text

    def test_pm_format(self) -> None:
        pattern = _make_pattern(start_hour=14)
        text = _build_spoken_text(pattern)
        assert "2pm" in text

    def test_noon(self) -> None:
        pattern = _make_pattern(start_hour=12)
        text = _build_spoken_text(pattern)
        assert "noon" in text

    def test_midnight(self) -> None:
        pattern = _make_pattern(start_hour=0)
        text = _build_spoken_text(pattern)
        assert "midnight" in text


# ---------------------------------------------------------------------------
# _pattern_key
# ---------------------------------------------------------------------------


class TestPatternKey:
    def test_uses_entity_and_service(self) -> None:
        pattern = _make_pattern(entity_id="light.kitchen", service="turn_on")
        key = _pattern_key(pattern)
        assert key == "light.kitchen:turn_on"

    def test_fallback_to_automation_string(self) -> None:
        # Pattern dict without direct entity_id / service fields
        pattern = {
            "pattern": "whatever",
            "frequency": 3,
            "suggested_automation": "Automate: turn_off light.bedroom_lamp daily at 22:00",
        }
        key = _pattern_key(pattern)
        assert "light.bedroom_lamp" in key
        assert "turn_off" in key


# ---------------------------------------------------------------------------
# SuggestionEngine.get_suggestion
# ---------------------------------------------------------------------------


class TestGetSuggestion:
    def test_returns_key_and_spoken_text(self, tmp_path: Path) -> None:
        engine = SuggestionEngine(
            dismissed_path=tmp_path / "dismissed.json",
            automations_path=tmp_path / "automations.json",
        )
        patterns = [_make_pattern()]
        result = engine.get_suggestion(patterns)
        assert result is not None
        key, spoken = result
        assert "light.kitchen_ceiling" in key
        assert "turn on" in spoken
        assert "Want me to automate that?" in spoken

    def test_at_most_one_suggestion_per_session(self, tmp_path: Path) -> None:
        engine = SuggestionEngine(
            dismissed_path=tmp_path / "dismissed.json",
            automations_path=tmp_path / "automations.json",
        )
        patterns = [_make_pattern(), _make_pattern(entity_id="light.bedroom")]
        engine.get_suggestion(patterns)
        # Second call in the same session must return None
        result2 = engine.get_suggestion(patterns)
        assert result2 is None

    def test_empty_patterns_returns_none(self, tmp_path: Path) -> None:
        engine = SuggestionEngine(
            dismissed_path=tmp_path / "dismissed.json",
            automations_path=tmp_path / "automations.json",
        )
        assert engine.get_suggestion([]) is None

    def test_sets_has_pending(self, tmp_path: Path) -> None:
        engine = SuggestionEngine(
            dismissed_path=tmp_path / "dismissed.json",
            automations_path=tmp_path / "automations.json",
        )
        assert not engine.has_pending
        engine.get_suggestion([_make_pattern()])
        assert engine.has_pending


# ---------------------------------------------------------------------------
# Accept flow
# ---------------------------------------------------------------------------


class TestAcceptFlow:
    def test_yes_saves_automation(self, tmp_path: Path) -> None:
        automations_path = tmp_path / "automations.json"
        engine = SuggestionEngine(
            dismissed_path=tmp_path / "dismissed.json",
            automations_path=automations_path,
        )
        patterns = [_make_pattern()]
        engine.get_suggestion(patterns)

        reply = engine.handle_yes()

        assert "set that up" in reply.lower() or "great" in reply.lower()
        assert automations_path.exists()
        saved = json.loads(automations_path.read_text())
        assert len(saved) == 1
        assert "light.kitchen_ceiling" in saved[0]["key"]

    def test_yes_clears_pending(self, tmp_path: Path) -> None:
        engine = SuggestionEngine(
            dismissed_path=tmp_path / "dismissed.json",
            automations_path=tmp_path / "automations.json",
        )
        engine.get_suggestion([_make_pattern()])
        engine.handle_yes()
        assert not engine.has_pending

    def test_is_accept_recognises_yes(self, tmp_path: Path) -> None:
        engine = SuggestionEngine(
            dismissed_path=tmp_path / "dismissed.json",
            automations_path=tmp_path / "automations.json",
        )
        assert engine.is_accept("yes")
        assert engine.is_accept("Yeah")
        assert engine.is_accept("Sure")
        assert not engine.is_accept("no thanks")


# ---------------------------------------------------------------------------
# Dismiss flow
# ---------------------------------------------------------------------------


class TestDismissFlow:
    def test_dismiss_records_to_disk(self, tmp_path: Path) -> None:
        dismissed_path = tmp_path / "dismissed.json"
        engine = SuggestionEngine(
            dismissed_path=dismissed_path,
            automations_path=tmp_path / "automations.json",
        )
        patterns = [_make_pattern()]
        engine.get_suggestion(patterns)

        reply = engine.handle_dismiss()

        assert "won't suggest" in reply.lower() or "got it" in reply.lower()
        assert dismissed_path.exists()
        data = json.loads(dismissed_path.read_text())
        assert any("light.kitchen_ceiling" in k for k in data)

    def test_dismissed_pattern_not_re_suggested(self, tmp_path: Path) -> None:
        dismissed_path = tmp_path / "dismissed.json"
        engine = SuggestionEngine(
            dismissed_path=dismissed_path,
            automations_path=tmp_path / "automations.json",
        )
        patterns = [_make_pattern()]
        engine.get_suggestion(patterns)
        engine.handle_dismiss()

        # New session (reset the session flag)
        engine._suggested_this_session = False
        result = engine.get_suggestion(patterns)
        assert result is None

    def test_dismissed_pattern_re_suggested_after_window(self, tmp_path: Path) -> None:
        dismissed_path = tmp_path / "dismissed.json"
        engine = SuggestionEngine(
            dismissed_path=dismissed_path,
            automations_path=tmp_path / "automations.json",
        )
        pattern = _make_pattern()
        key = _pattern_key(pattern)

        # Write a dismissal timestamp 31 days in the past
        old_ts = time.time() - (31 * 86400)
        dismissed_path.parent.mkdir(parents=True, exist_ok=True)
        dismissed_path.write_text(json.dumps({key: old_ts}))

        result = engine.get_suggestion([pattern])
        assert result is not None

    def test_is_dismiss_recognises_no_thanks(self, tmp_path: Path) -> None:
        engine = SuggestionEngine(
            dismissed_path=tmp_path / "dismissed.json",
            automations_path=tmp_path / "automations.json",
        )
        assert engine.is_dismiss("no thanks")
        assert engine.is_dismiss("No Thanks")
        assert engine.is_dismiss("nope")
        assert engine.is_dismiss("not now")
        assert not engine.is_dismiss("yes")

    def test_dismiss_clears_pending(self, tmp_path: Path) -> None:
        engine = SuggestionEngine(
            dismissed_path=tmp_path / "dismissed.json",
            automations_path=tmp_path / "automations.json",
        )
        engine.get_suggestion([_make_pattern()])
        engine.handle_dismiss()
        assert not engine.has_pending


# ---------------------------------------------------------------------------
# Integration with detect_patterns
# ---------------------------------------------------------------------------


class TestPatternDetectorIntegration:
    def test_detect_patterns_returns_new_fields(self) -> None:
        """detect_patterns result dicts must include entity_id, service, start_hour."""
        from datetime import date

        # Use local midnight so datetime.fromtimestamp() yields consistent hours
        midnight = time.mktime(date.today().timetuple())
        entries = [
            PatternEntry(
                entity_id="light.kitchen_ceiling",
                service="turn_on",
                timestamp=midnight + 7 * 3600 + i * 86400,
            )
            for i in range(3)
        ]
        patterns = detect_patterns(entries, min_occurrences=3, time_window_hours=2.0)
        assert len(patterns) >= 1
        p = patterns[0]
        assert p["entity_id"] == "light.kitchen_ceiling"
        assert p["service"] == "turn_on"
        assert "start_hour" in p

    def test_engine_suggest_from_detected_patterns(self, tmp_path: Path) -> None:
        from datetime import date

        # Use local midnight so datetime.fromtimestamp() yields the expected hour
        midnight = time.mktime(date.today().timetuple())
        entries = [
            PatternEntry(
                entity_id="light.kitchen_ceiling",
                service="turn_on",
                timestamp=midnight + 7 * 3600 + i * 86400,
            )
            for i in range(3)
        ]
        patterns = detect_patterns(entries, min_occurrences=3, time_window_hours=1.0)
        engine = SuggestionEngine(
            dismissed_path=tmp_path / "dismissed.json",
            automations_path=tmp_path / "automations.json",
        )
        result = engine.get_suggestion(patterns)
        assert result is not None
        _, spoken = result
        assert "kitchen ceiling" in spoken
        assert "7am" in spoken
