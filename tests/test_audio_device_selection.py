import builtins
import json

import pytest

pytest.importorskip("sounddevice")

from utils import audio_device


class DummySoundDevice:
    def __init__(self) -> None:
        self._devices = [
            {
                "name": "Mic 0",
                "max_input_channels": 2,
                "default_samplerate": 16000,
                "hostapi": 0,
            },
            {
                "name": "Mic 1",
                "max_input_channels": 2,
                "default_samplerate": 16000,
                "hostapi": 0,
            },
            {
                "name": "Mic 2",
                "max_input_channels": 2,
                "default_samplerate": 16000,
                "hostapi": 0,
            },
        ]

    def query_devices(self, index=None):
        if index is None:
            return self._devices
        return self._devices[index]

    def query_hostapis(self, index):
        return {"name": "WASAPI"}


def test_resolve_audio_device_uses_rex_config(monkeypatch, tmp_path):
    config_path = tmp_path / "config" / "rex_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "audio": {
                    "input_device_index": 2,
                    "sample_rate": 16000,
                }
            }
        )
    )

    dummy_sd = DummySoundDevice()
    monkeypatch.setattr(audio_device, "sd", dummy_sd, raising=False)
    monkeypatch.setattr(audio_device, "DEFAULT_REX_CONFIG_PATH", config_path)

    real_open = builtins.open

    def guarded_open(path, *args, **kwargs):
        if str(path).endswith("audio_config.json"):
            raise AssertionError("Legacy audio_config.json should not be accessed")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)

    audio_config = audio_device.load_audio_config()
    resolved_device, status_msg = audio_device.resolve_audio_device(
        audio_config["input_device_index"],
        audio_config["sample_rate"],
    )

    assert resolved_device == 2
    assert "2:" in status_msg
