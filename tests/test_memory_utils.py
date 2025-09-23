"""Tests for the memory utility helpers."""

from __future__ import annotations

from memory_utils import (
    extract_voice_reference,
    load_all_profiles,
    load_users_map,
    resolve_user_key,
)


def test_resolve_user_key_from_email():
    users_map = load_users_map()
    profiles = load_all_profiles()

    result = resolve_user_key("james@example.com", users_map, profiles=profiles)
    assert result == "james"


def test_extract_voice_reference_handles_missing(monkeypatch, tmp_path):
    profile = {"voice": {"sample_path": str(tmp_path / "voice.wav")}}
    (tmp_path / "voice.wav").write_bytes(b"fake")

    assert extract_voice_reference(profile) == str(tmp_path / "voice.wav")

    profile_missing = {"voice": {}}
    assert extract_voice_reference(profile_missing) is None
