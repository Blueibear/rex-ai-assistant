"""Tests for US-036: Surface proactive suggestions to the user.

Covers:
- At most one suggestion per user per session
- Spoken text format
- Accept flow (yes → automation saved, tagged with the accepting user)
- Dismiss flow (no thanks → dismissal recorded per user, not re-suggested for 30 days)
- Already-dismissed pattern is skipped
- Legacy flat dismissed-file format is attributed to the "default" user
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


def _make_engine(tmp_path: Path) -> SuggestionEngine:
    return SuggestionEngine(
        dismissed_path=tmp_path / "dismissed.json",
        automations_path=tmp_path / "automations.json",
    )


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
        engine = _make_engine(tmp_path)
        patterns = [_make_pattern()]
        result = engine.get_suggestion(patterns, "alice")
        assert result is not None
        key, spoken = result
        assert "light.kitchen_ceiling" in key
        assert "turn on" in spoken
        assert "Want me to automate that?" in spoken

    def test_at_most_one_suggestion_per_session(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        patterns = [_make_pattern(), _make_pattern(entity_id="light.bedroom")]
        engine.get_suggestion(patterns, "alice")
        # Second call in the same session must return None for the same user
        result2 = engine.get_suggestion(patterns, "alice")
        assert result2 is None

    def test_empty_patterns_returns_none(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        assert engine.get_suggestion([], "alice") is None

    def test_sets_has_pending(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        assert not engine.has_pending("alice")
        engine.get_suggestion([_make_pattern()], "alice")
        assert engine.has_pending("alice")

    def test_missing_user_returns_none(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        assert engine.get_suggestion([_make_pattern()], None) is None
        assert engine.get_suggestion([_make_pattern()], "") is None

    def test_invalid_user_returns_none(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        assert engine.get_suggestion([_make_pattern()], "../evil") is None
        assert not engine.has_pending("../evil")


# ---------------------------------------------------------------------------
# Accept flow
# ---------------------------------------------------------------------------


class TestAcceptFlow:
    def test_yes_saves_automation(self, tmp_path: Path) -> None:
        automations_path = tmp_path / "automations.json"
        engine = _make_engine(tmp_path)
        patterns = [_make_pattern()]
        engine.get_suggestion(patterns, "alice")

        reply = engine.handle_yes("alice")

        assert "set that up" in reply.lower() or "great" in reply.lower()
        assert automations_path.exists()
        saved = json.loads(automations_path.read_text())
        assert len(saved) == 1
        assert "light.kitchen_ceiling" in saved[0]["key"]
        assert saved[0]["user_id"] == "alice"

    def test_yes_clears_pending(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.get_suggestion([_make_pattern()], "alice")
        engine.handle_yes("alice")
        assert not engine.has_pending("alice")

    def test_is_accept_recognises_yes(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
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
        engine = _make_engine(tmp_path)
        patterns = [_make_pattern()]
        engine.get_suggestion(patterns, "alice")

        reply = engine.handle_dismiss("alice")

        assert "won't suggest" in reply.lower() or "got it" in reply.lower()
        assert dismissed_path.exists()
        data = json.loads(dismissed_path.read_text())
        assert any("light.kitchen_ceiling" in k for k in data["users"]["alice"])

    def test_dismissed_pattern_not_re_suggested(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        patterns = [_make_pattern()]
        engine.get_suggestion(patterns, "alice")
        engine.handle_dismiss("alice")

        # New session (reset the per-user session flag)
        engine.reset_session("alice")
        result = engine.get_suggestion(patterns, "alice")
        assert result is None

    def test_dismissed_pattern_re_suggested_after_window(self, tmp_path: Path) -> None:
        dismissed_path = tmp_path / "dismissed.json"
        engine = _make_engine(tmp_path)
        pattern = _make_pattern()
        key = _pattern_key(pattern)

        # Write a dismissal timestamp 31 days in the past
        old_ts = time.time() - (31 * 86400)
        dismissed_path.parent.mkdir(parents=True, exist_ok=True)
        dismissed_path.write_text(json.dumps({"users": {"alice": {key: old_ts}}}))

        result = engine.get_suggestion([pattern], "alice")
        assert result is not None

    def test_legacy_flat_dismissed_file_maps_to_default_user(self, tmp_path: Path) -> None:
        """Pre-per-user flat files belong to "default", not to everyone."""
        dismissed_path = tmp_path / "dismissed.json"
        engine = _make_engine(tmp_path)
        pattern = _make_pattern()
        key = _pattern_key(pattern)

        dismissed_path.parent.mkdir(parents=True, exist_ok=True)
        dismissed_path.write_text(json.dumps({key: time.time()}))

        # The legacy dismissal suppresses the suggestion for "default" ...
        assert engine.get_suggestion([pattern], "default") is None
        # ... but not for other users
        assert engine.get_suggestion([pattern], "alice") is not None

    def test_is_dismiss_recognises_no_thanks(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        assert engine.is_dismiss("no thanks")
        assert engine.is_dismiss("No Thanks")
        assert engine.is_dismiss("nope")
        assert engine.is_dismiss("not now")
        assert not engine.is_dismiss("yes")

    def test_dismiss_clears_pending(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.get_suggestion([_make_pattern()], "alice")
        engine.handle_dismiss("alice")
        assert not engine.has_pending("alice")


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
        engine = _make_engine(tmp_path)
        result = engine.get_suggestion(patterns, "alice")
        assert result is not None
        _, spoken = result
        assert "kitchen ceiling" in spoken
        assert "7am" in spoken


# ---------------------------------------------------------------------------
# Contextual proactive candidates (US-123)
# ---------------------------------------------------------------------------


def _make_contextual_candidate(user_id: str = "alice"):
    from rex.proactivity.models import ProactiveCandidate

    return ProactiveCandidate(
        key="commute:weather-delay",
        user_id=user_id,
        spoken_text="Traffic and storms could slow you down. Leaving 20 minutes early would help.",
        source_ids=("integration:calendar", "integration:traffic", "integration:weather"),
        freshness_seconds=60.0,
        confidence=0.9,
        benefit=0.9,
        urgency=0.85,
        suggested_action="show_route",
    )


def test_contextual_suggestion_reuses_session_and_pending_state(tmp_path: Path) -> None:
    engine = _make_engine(tmp_path)
    candidate = _make_contextual_candidate()

    result = engine.get_contextual_suggestion([candidate], user_id="alice")

    assert result == (candidate.key, candidate.spoken_text)
    assert engine.has_pending("alice")
    assert engine.get_contextual_suggestion([candidate], user_id="alice") is None


def test_contextual_suggestion_rejects_foreign_candidate(tmp_path: Path) -> None:
    engine = _make_engine(tmp_path)

    assert (
        engine.get_contextual_suggestion(
            [_make_contextual_candidate(user_id="bob")], user_id="alice"
        )
        is None
    )


def test_contextual_acceptance_does_not_create_automation(tmp_path: Path) -> None:
    engine = _make_engine(tmp_path)
    candidate = _make_contextual_candidate()
    engine.get_contextual_suggestion([candidate], user_id="alice")

    reply = engine.handle_yes("alice")

    assert "noted" in reply.lower() or "okay" in reply.lower()
    assert not (tmp_path / "automations.json").exists()
    assert not engine.has_pending("alice")
