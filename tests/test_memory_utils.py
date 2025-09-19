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
    absolute_voice = tmp_path / "voice.wav"
    absolute_voice.write_bytes(b"fake")
    profile = {"voice": {"sample_path": str(absolute_voice)}}

    assert extract_voice_reference(profile) == str(absolute_voice.resolve())

    user_dir = tmp_path / "james"
    user_dir.mkdir()
    relative_voice = user_dir / "voice.wav"
    relative_voice.write_bytes(b"demo")
    profile_relative = {"voice_sample": "voice.wav"}

    assert extract_voice_reference(
        profile_relative,
        user_key="james",
        memory_root=str(tmp_path),
    ) == str(relative_voice.resolve())

    profile_missing = {"voice": {}}
    assert (
        extract_voice_reference(
            profile_missing,
            user_key="james",
            memory_root=str(tmp_path),
        )
        is None
    )
