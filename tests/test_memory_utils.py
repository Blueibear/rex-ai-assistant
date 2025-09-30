"""Tests for the memory utility helpers."""

from __future__ import annotations

import types

import memory_utils
from memory_utils import (
    append_history_entry,
    export_transcript,
    extract_voice_reference,
    load_all_profiles,
    load_recent_history,
    load_users_map,
    resolve_user_key,
    trim_history,
)


def test_resolve_user_key_from_email():
    users_map = load_users_map()
    profiles = load_all_profiles()

    result = resolve_user_key("james@example.com", users_map, profiles=profiles)
    assert result == "james"


def test_extract_voice_reference_handles_missing(tmp_path):
    voice_file = tmp_path / "voice.wav"
    voice_file.write_bytes(b"fake")

    profile_with_voice = {"voice": {"sample_path": str(voice_file)}}
    assert extract_voice_reference(profile_with_voice) == str(voice_file)

    profile_missing = {"voice": {}}
    assert extract_voice_reference(profile_missing) is None

    profile_none = {}
    assert extract_voice_reference(profile_none) is None


def test_trim_history_limits_entries():
    history = [{"id": i} for i in range(10)]

    trimmed = trim_history(history, limit=3)

    assert len(trimmed) == 3
    assert trimmed[0]["id"] == 7
    assert trimmed[-1]["id"] == 9


def test_append_history_trims(monkeypatch, tmp_path):
    monkeypatch.setattr(
        memory_utils,
        "rex_settings",
        types.SimpleNamespace(max_memory_items=2),
        raising=False,
    )

    for idx in range(4):
        append_history_entry(
            "tester",
            {"role": "user", "text": f"utterance {idx}"},
            memory_root=tmp_path,
        )

    entries = load_recent_history("tester", memory_root=tmp_path)
    assert len(entries) == 2
    assert entries[0]["text"] == "utterance 2"
    assert entries[1]["text"] == "utterance 3"


def test_export_transcript_appends(monkeypatch, tmp_path):
    transcripts_dir = tmp_path / "transcripts"
    monkeypatch.setattr(
        memory_utils,
        "rex_settings",
        types.SimpleNamespace(transcripts_dir=str(transcripts_dir)),
        raising=False,
    )

    conversation = [
        {"role": "user", "text": "Hello"},
        {"role": "assistant", "text": "Hi there"},
    ]

    path = export_transcript("tester", conversation, transcripts_dir=transcripts_dir)
    assert path.exists()
    contents = path.read_text(encoding="utf-8")
    assert "user: Hello" in contents
    assert "assistant: Hi there" in contents


def test_export_transcript_creates_directory(monkeypatch, tmp_path):
    transcripts_dir = tmp_path / "not_yet_there"
    monkeypatch.setattr(
        memory_utils,
        "rex_settings",
        types.SimpleNamespace(transcripts_dir=str(transcripts_dir)),
        raising=False,
    )

    conversation = [{"role": "assistant", "text": "Testing export..."}]
    path = export_transcript("tester", conversation, transcripts_dir=transcripts_dir)

    assert path.exists()
    assert path.parent.exists()


def test_load_recent_history_with_missing_file(tmp_path):
    entries = load_recent_history("ghost", memory_root=tmp_path)
    assert isinstance(entries, list)
    assert len(entries) == 0
