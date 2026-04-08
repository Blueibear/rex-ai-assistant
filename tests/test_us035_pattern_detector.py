"""Tests for US-035: proactive suggestion engine (pattern detection)."""

from __future__ import annotations

from rex.suggestions.pattern_detector import PatternEntry, detect_patterns


def _make_entries(
    entity_id: str,
    service: str,
    hour: int,
    count: int,
    base_ts: float | None = None,
) -> list[PatternEntry]:
    """Create *count* PatternEntry objects at the given hour of day."""
    if base_ts is None:
        # Use a fixed reference: 2024-01-01 00:00:00 UTC as midnight
        import calendar
        import datetime

        midnight = calendar.timegm(datetime.date(2024, 1, 1).timetuple())
        base_ts = float(midnight)

    entries = []
    for i in range(count):
        # Each entry is on a different day at the same hour
        ts = base_ts + i * 86400 + hour * 3600
        entries.append(PatternEntry(entity_id=entity_id, service=service, timestamp=ts))
    return entries


# ---------------------------------------------------------------------------
# Basic pattern detection
# ---------------------------------------------------------------------------


def test_detect_patterns_returns_pattern_with_enough_occurrences() -> None:
    """3 or more occurrences should produce a pattern entry."""
    entries = _make_entries("light.kitchen", "turn_on", hour=7, count=3)
    patterns = detect_patterns(entries)
    assert len(patterns) == 1
    assert patterns[0]["frequency"] == 3


def test_detect_patterns_below_threshold_not_returned() -> None:
    """Fewer than 3 occurrences must NOT produce a pattern."""
    entries = _make_entries("light.kitchen", "turn_on", hour=7, count=2)
    patterns = detect_patterns(entries, min_occurrences=3)
    assert patterns == []


def test_detect_patterns_exact_threshold() -> None:
    """Exactly min_occurrences should still qualify."""
    entries = _make_entries("switch.fan", "turn_on", hour=8, count=5)
    patterns = detect_patterns(entries, min_occurrences=5)
    assert len(patterns) == 1
    assert patterns[0]["frequency"] == 5


def test_detect_patterns_returns_required_keys() -> None:
    """Each pattern dict must have pattern, frequency, and suggested_automation keys."""
    entries = _make_entries("light.bedroom", "turn_off", hour=22, count=4)
    patterns = detect_patterns(entries)
    assert len(patterns) == 1
    p = patterns[0]
    assert "pattern" in p
    assert "frequency" in p
    assert "suggested_automation" in p


def test_detect_patterns_pattern_string_is_human_readable() -> None:
    """Pattern string should mention the entity name and service."""
    entries = _make_entries("light.kitchen_ceiling", "turn_on", hour=7, count=3)
    patterns = detect_patterns(entries)
    assert len(patterns) == 1
    pattern_str = patterns[0]["pattern"]
    assert "kitchen ceiling" in pattern_str
    assert "turn on" in pattern_str


def test_detect_patterns_suggested_automation_references_entity() -> None:
    """Suggested automation string should reference the entity and service."""
    entries = _make_entries("light.hallway", "turn_on", hour=6, count=3)
    patterns = detect_patterns(entries)
    automation = patterns[0]["suggested_automation"]
    assert "light.hallway" in automation
    assert "turn_on" in automation


# ---------------------------------------------------------------------------
# Multiple patterns, sorting
# ---------------------------------------------------------------------------


def test_detect_patterns_sorted_by_frequency_descending() -> None:
    """Patterns must be sorted with highest frequency first."""
    entries = _make_entries("light.kitchen", "turn_on", hour=7, count=5) + _make_entries(
        "switch.fan", "turn_on", hour=8, count=3
    )
    patterns = detect_patterns(entries)
    assert len(patterns) == 2
    assert patterns[0]["frequency"] >= patterns[1]["frequency"]


def test_detect_patterns_multiple_entities_independent() -> None:
    """Patterns for different entities should be returned independently."""
    entries = _make_entries("light.kitchen", "turn_on", hour=7, count=4) + _make_entries(
        "light.bedroom", "turn_off", hour=22, count=3
    )
    patterns = detect_patterns(entries)
    entity_ids = {p["suggested_automation"].split()[2] for p in patterns}
    assert "light.kitchen" in entity_ids
    assert "light.bedroom" in entity_ids


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_detect_patterns_empty_history_returns_empty() -> None:
    """Empty input should return an empty list."""
    assert detect_patterns([]) == []


def test_detect_patterns_suggestions_not_acted_on_automatically() -> None:
    """detect_patterns() must return data only — it must not have side-effects.

    We verify this by checking that calling it twice with the same history
    produces the same output (no mutation, no external calls).
    """
    entries = _make_entries("light.living_room", "turn_on", hour=9, count=3)
    first = detect_patterns(entries)
    second = detect_patterns(entries)
    assert first == second
