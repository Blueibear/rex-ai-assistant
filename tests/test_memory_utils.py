from types import SimpleNamespace

from types import SimpleNamespace

from rex.memory import (
    append_history_entry,
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
    monkeypatch.setattr(
        "rex.memory.settings",
        SimpleNamespace(memory_max_turns=2),
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
