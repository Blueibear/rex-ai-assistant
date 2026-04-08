"""Tests for US-025: device alias system with synonym and fuzzy matching."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rex.ha.device_aliases import AliasResolver


@pytest.fixture()
def alias_file(tmp_path: Path) -> Path:
    data = {
        "aliases": {
            "bedroom light": "light.bedroom_main",
            "kitchen light": "light.kitchen_ceiling",
            "living room tv": "media_player.lounge_tv",
            "front door lock": "lock.front_door",
        },
        "synonyms": {
            "lamp": "light",
            "telly": "tv",
        },
    }
    p = tmp_path / "device_aliases.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture()
def resolver(alias_file: Path) -> AliasResolver:
    return AliasResolver(aliases_path=alias_file)


# ------------------------------------------------------------------
# Exact match
# ------------------------------------------------------------------


def test_exact_match_returns_entity_and_full_confidence(resolver: AliasResolver) -> None:
    result = resolver.resolve("bedroom light")
    assert result is not None
    entity_id, confidence = result
    assert entity_id == "light.bedroom_main"
    assert confidence == 1.0


def test_exact_match_case_insensitive(resolver: AliasResolver) -> None:
    result = resolver.resolve("Bedroom Light")
    assert result is not None
    assert result[0] == "light.bedroom_main"
    assert result[1] == 1.0


# ------------------------------------------------------------------
# Fuzzy match
# ------------------------------------------------------------------


def test_fuzzy_match_one_typo(resolver: AliasResolver) -> None:
    # "bedrom light" has 1 edit vs "bedroom light"
    result = resolver.resolve("bedrom light")
    assert result is not None
    entity_id, confidence = result
    assert entity_id == "light.bedroom_main"
    assert 0.0 < confidence < 1.0


def test_fuzzy_match_two_edits(resolver: AliasResolver) -> None:
    # "bedrm ligt" has 2 edits vs "bedroom light"
    result = resolver.resolve("bedrm light")
    assert result is not None
    assert result[0] == "light.bedroom_main"


def test_fuzzy_no_match_beyond_threshold(resolver: AliasResolver) -> None:
    # "xyz abc def" is very far from any alias
    result = resolver.resolve("xyz abc def")
    assert result is None


# ------------------------------------------------------------------
# Synonym match
# ------------------------------------------------------------------


def test_synonym_lamp_resolves_to_light(resolver: AliasResolver) -> None:
    # "bedroom lamp" -> expand "lamp"->"light" -> "bedroom light"
    result = resolver.resolve("bedroom lamp")
    assert result is not None
    entity_id, confidence = result
    assert entity_id == "light.bedroom_main"
    assert confidence < 1.0  # synonym expansion lowers confidence slightly


def test_synonym_telly_resolves_to_tv(resolver: AliasResolver) -> None:
    # "living room telly" -> "living room tv"
    result = resolver.resolve("living room telly")
    assert result is not None
    assert result[0] == "media_player.lounge_tv"


# ------------------------------------------------------------------
# No-match cases
# ------------------------------------------------------------------


def test_no_match_returns_none(resolver: AliasResolver) -> None:
    result = resolver.resolve("garage opener")
    assert result is None


def test_empty_aliases_file_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "device_aliases.json"
    p.write_text(json.dumps({"aliases": {}, "synonyms": {}}), encoding="utf-8")
    r = AliasResolver(aliases_path=p)
    assert r.resolve("bedroom light") is None


def test_missing_aliases_file_returns_none(tmp_path: Path) -> None:
    r = AliasResolver(aliases_path=tmp_path / "nonexistent.json")
    assert r.resolve("bedroom light") is None
