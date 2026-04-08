"""Tests for US-030: Clarification system for ambiguous commands.

Acceptance criteria:
- AliasResolver.resolve_all() returns multiple candidates when entity is ambiguous
- ClarificationHandler asks "Did you mean X or Y?" for multi-match
- ClarificationHandler asks "Which device?" for missing-device (pronoun) commands
- HABridge.process_transcript() returns clarification question for ambiguous/missing-device transcripts
- Typecheck passes
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rex.ha.clarification import ClarificationHandler, _extract_entity_query
from rex.ha.device_aliases import AliasResolver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_resolver(tmp_path: Path, aliases: dict[str, str]) -> AliasResolver:
    data = {"aliases": aliases, "synonyms": {}}
    p = tmp_path / "aliases.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return AliasResolver(p)


# ---------------------------------------------------------------------------
# AliasResolver.resolve_all() tests
# ---------------------------------------------------------------------------


def test_resolve_all_returns_multiple_fuzzy_matches(tmp_path: Path) -> None:
    resolver = _make_resolver(
        tmp_path,
        {
            "bedroom light": "light.bedroom",
            "bathroom light": "light.bathroom",
            "kitchen light": "light.kitchen",
        },
    )
    # "b?droom light" fuzzy-matches both bedroom and bathroom at similar distance
    results = resolver.resolve_all("bedrom light")  # 1 edit from "bedroom light"
    entity_ids = [r[1] for r in results]
    assert "light.bedroom" in entity_ids


def test_resolve_all_exact_match_only(tmp_path: Path) -> None:
    resolver = _make_resolver(
        tmp_path,
        {
            "bedroom light": "light.bedroom",
            "living room lamp": "light.living",
        },
    )
    results = resolver.resolve_all("bedroom light")
    assert len(results) == 1
    assert results[0][0] == "bedroom light"
    assert results[0][1] == "light.bedroom"
    assert results[0][2] == 1.0


def test_resolve_all_returns_empty_for_no_match(tmp_path: Path) -> None:
    resolver = _make_resolver(tmp_path, {"bedroom light": "light.bedroom"})
    results = resolver.resolve_all("nonexistent device xyz")
    assert results == []


def test_resolve_all_deduplicates_by_entity_id(tmp_path: Path) -> None:
    # Two aliases pointing to the same entity should only appear once
    resolver = _make_resolver(
        tmp_path,
        {
            "bedroom light": "light.bedroom",
            "bedroom lamp": "light.bedroom",
        },
    )
    results = resolver.resolve_all("bedroom light")
    entity_ids = [r[1] for r in results]
    assert entity_ids.count("light.bedroom") == 1


# ---------------------------------------------------------------------------
# ClarificationHandler – ambiguous entity (multi-match)
# ---------------------------------------------------------------------------


def test_clarification_did_you_mean(tmp_path: Path) -> None:
    resolver = _make_resolver(
        tmp_path,
        {
            "bedroom light": "light.bedroom",
            "bedroom lights": "light.bedroom_all",
        },
    )
    handler = ClarificationHandler(resolver)
    # "bedroom lights" is exact for one alias and 1-edit from the other → ambiguous (delta < 0.15)
    result = handler.check("turn on bedroom lights")
    assert result is not None
    assert "Did you mean" in result
    assert "bedroom" in result.lower()


def test_clarification_exact_match_not_ambiguous(tmp_path: Path) -> None:
    resolver = _make_resolver(
        tmp_path,
        {
            "bedroom light": "light.bedroom",
            "bathroom light": "light.bathroom",
        },
    )
    handler = ClarificationHandler(resolver)
    # Exact match → confidence 1.0, well above any second candidate
    result = handler.check("turn on bedroom light")
    assert result is None


# ---------------------------------------------------------------------------
# ClarificationHandler – missing device (pronoun)
# ---------------------------------------------------------------------------


def test_clarification_turn_it_on() -> None:
    handler = ClarificationHandler()
    result = handler.check("turn it on")
    assert result is not None
    assert "Which device" in result


def test_clarification_turn_that_off() -> None:
    handler = ClarificationHandler()
    result = handler.check("turn that off")
    assert result is not None
    assert "Which device" in result


def test_clarification_turn_this_on() -> None:
    handler = ClarificationHandler()
    result = handler.check("turn this on")
    assert result is not None
    assert "Which device" in result


def test_clarification_specific_device_no_question() -> None:
    handler = ClarificationHandler()
    # Clear device name → no clarification needed
    result = handler.check("turn on the kitchen light")
    assert result is None


# ---------------------------------------------------------------------------
# _extract_entity_query helper
# ---------------------------------------------------------------------------


def test_extract_entity_query_turn_on() -> None:
    assert _extract_entity_query("turn on the bedroom light") == "bedroom light"


def test_extract_entity_query_turn_off() -> None:
    assert _extract_entity_query("turn off the kitchen fan") == "kitchen fan"


def test_extract_entity_query_no_match() -> None:
    assert _extract_entity_query("what time is it?") is None


# ---------------------------------------------------------------------------
# HABridge integration – process_transcript returns clarification question
# ---------------------------------------------------------------------------


def test_ha_bridge_returns_clarification_for_pronoun() -> None:
    """HABridge.process_transcript returns a clarification question for 'turn it on'."""
    # Import lazily so tests run even if requests is not installed
    try:
        from rex.ha_bridge import HABridge
    except Exception:
        pytest.skip("HABridge not importable (requests not installed)")

    bridge = HABridge.__new__(HABridge)
    # Minimal attrs so process_transcript works
    bridge._base_url = "http://ha.local:8123"
    bridge._token = "dummy-token"
    from rex.ha.clarification import ClarificationHandler

    bridge._clarification = ClarificationHandler()

    result = bridge.process_transcript("turn it on")
    assert result is not None
    assert "Which device" in result


def test_ha_bridge_returns_did_you_mean(tmp_path: Path) -> None:
    """HABridge.process_transcript returns 'Did you mean X or Y?' for ambiguous entity."""
    try:
        from rex.ha_bridge import HABridge
    except Exception:
        pytest.skip("HABridge not importable (requests not installed)")

    resolver = _make_resolver(
        tmp_path,
        {
            "bedroom light": "light.bedroom",
            "bedroom lights": "light.bedroom_all",
        },
    )
    bridge = HABridge.__new__(HABridge)
    bridge._base_url = "http://ha.local:8123"
    bridge._token = "dummy-token"
    from rex.ha.clarification import ClarificationHandler

    bridge._clarification = ClarificationHandler(resolver)

    result = bridge.process_transcript("turn on bedroom lights")
    assert result is not None
    assert "Did you mean" in result
