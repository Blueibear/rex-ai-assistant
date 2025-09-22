"""Tests for the memory utility helpers."""

from __future__ import annotations

import config
from memory_utils import (
    append_history_entry,
    export_transcript,
    extract_voice_reference,
    load_all_profiles,
    load_recent_history,
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


def test_append_history_trims(monkeypatch, tmp_path):
    cfg = config.AppConfig(memory_max_turns=2, transcripts_dir=tmp_path / "transcripts")
    monkeypatch.setattr(config, "_cached_config", cfg, raising=False)

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
    cfg = config.AppConfig(transcripts_dir=transcripts_dir)
    monkeypatch.setattr(config, "_cached_config", cfg, raising=False)

    conversation = [
        {"role": "user", "text": "Hello"},
        {"role": "assistant", "text": "Hi there"},
    ]

    path = export_transcript("tester", conversation, transcripts_dir=transcripts_dir)
    assert path.exists()
    contents = path.read_text(encoding="utf-8")
    assert "user: Hello" in contents
    assert "assistant: Hi there" in contents
