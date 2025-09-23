import audio_config
import pytest


def test_list_devices_requires_sounddevice(monkeypatch):
    monkeypatch.setattr(audio_config, "sd", None, raising=False)
    with pytest.raises(RuntimeError):
        audio_config.list_devices()


def test_main_updates_env(monkeypatch):
    updated = {}

    def fake_update(key, value):
        updated[key] = value

    monkeypatch.setattr(audio_config, "update_env_value", fake_update)
    monkeypatch.setattr(audio_config, "reload_settings", lambda: None)

    exit_code = audio_config.main(["--set-input", "1", "--set-output", "2"])

    assert exit_code == 0
    assert updated["REX_INPUT_DEVICE"] == "1"
    assert updated["REX_OUTPUT_DEVICE"] == "2"
