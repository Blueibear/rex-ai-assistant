import audio_config
import pytest


class _DummyStream:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _DummySoundDevice:
    def __init__(self):
        self.devices = [
            {"name": "Mic", "max_input_channels": 2, "max_output_channels": 0},
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
        ]

    def query_devices(self):
        return self.devices

    def InputStream(self, *args, **kwargs):  # noqa: N802 - mimic sounddevice API
        return _DummyStream()

    def OutputStream(self, *args, **kwargs):  # noqa: N802 - mimic sounddevice API
        return _DummyStream()


def test_list_devices_requires_sounddevice(monkeypatch):
    monkeypatch.setattr(audio_config, "sd", None, raising=False)
    with pytest.raises(audio_config.AudioDeviceError):
        audio_config.list_devices()


def test_main_updates_env(monkeypatch):
    updated = {}

    def fake_update(key, value):
        updated[key] = value

    monkeypatch.setattr(audio_config, "sd", _DummySoundDevice(), raising=False)
    monkeypatch.setattr(audio_config, "update_env_value", fake_update)

    exit_code = audio_config.main(["--set-input", "0", "--set-output", "1"])

    assert exit_code == 0
    assert updated["REX_INPUT_DEVICE"] == "0"
    assert updated["REX_OUTPUT_DEVICE"] == "1"
