"""Behavior tests for the canonical Electron setup bridge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bridge import rex_setup_bridge
from rex.assistant_errors import AudioDeviceError


def _install_setup_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], list[str]]:
    captured: dict[str, Any] = {}
    bootstrapped: list[str] = []

    def fake_create_user(username: str, password: str) -> dict[str, str]:
        assert username == "james"
        assert password == "securepass1"  # pragma: allowlist secret
        return {"id": "james"}

    def fake_persist(
        values: dict[str, str], *, config_path=None, update_config=None
    ) -> dict[str, str]:
        captured["values"] = dict(values)
        config: dict[str, Any] = {}
        if update_config is not None:
            update_config(config)
        captured["config"] = config
        return {}

    monkeypatch.setattr("rex.auth.create_user", fake_create_user)
    monkeypatch.setattr("rex.permissions.bootstrap_admin_if_first_user", bootstrapped.append)
    monkeypatch.setattr("rex.credential_persistence.persist_household_secrets", fake_persist)
    return captured, bootstrapped


def test_deferred_home_assistant_is_omitted_from_persistence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured, bootstrapped = _install_setup_fakes(monkeypatch)

    rex_setup_bridge._handle_complete(
        {
            "username": "james",
            "password": "securepass1",  # pragma: allowlist secret
            "llm_provider": "local",
            "tts_provider": "none",
            "ha_base_url": "http://ha.local:8123",
            "ha_token": "must-not-persist",  # pragma: allowlist secret
            "defer_home_assistant": True,
        }
    )

    assert json.loads(capsys.readouterr().out) == {"ok": True, "user_id": "james"}
    assert captured["values"] == {}
    assert "home_assistant" not in captured["config"]
    assert bootstrapped == ["james"]


def test_string_false_does_not_defer_home_assistant(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured, _ = _install_setup_fakes(monkeypatch)

    rex_setup_bridge._handle_complete(
        {
            "username": "james",
            "password": "securepass1",  # pragma: allowlist secret
            "llm_provider": "local",
            "tts_provider": "none",
            "ha_base_url": "http://ha.local:8123",
            "ha_token": "test-token",  # pragma: allowlist secret
            "defer_home_assistant": "false",
        }
    )

    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert captured["values"] == {"HA_TOKEN": "test-token"}  # pragma: allowlist secret
    assert captured["config"]["home_assistant"]["base_url"] == "http://ha.local:8123"


def test_household_voice_choices_persist_to_canonical_runtime_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured, _ = _install_setup_fakes(monkeypatch)

    rex_setup_bridge._handle_complete(
        {
            "username": "james",
            "password": "securepass1",  # pragma: allowlist secret
            "llm_provider": "local",
            "tts_provider": "edge",
            "tts_voice_id": "en-US-AriaNeural",
            "microphone_device_index": 2,
            "speaker_device_index": 4,
            "wake_word_id": "hey_rex",
            "local_device_id": "local_voice",
            "room_name": "Office",
            "background_voice_enabled": True,
            "ha_base_url": "",
            "ha_token": "",
        }
    )

    assert json.loads(capsys.readouterr().out)["ok"] is True
    config = captured["config"]
    assert config["models"]["tts_provider"] == "edge"
    assert config["models"]["tts_voice"] == "en-US-AriaNeural"
    assert config["audio"]["input_device_index"] == 2
    assert config["audio"]["output_device_index"] == 4
    assert config["wakeword"]["wakeword"] == "hey_rex"
    assert config["device_room_map"] == {"local_voice": "Office"}
    assert config["runtime"]["active_user"] == "james"
    assert config["runtime"]["background_voice_enabled"] is True


def test_background_voice_defaults_off_in_setup_runtime_contract(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured, _ = _install_setup_fakes(monkeypatch)

    rex_setup_bridge._handle_complete(
        {
            "username": "james",
            "password": "securepass1",  # pragma: allowlist secret
            "llm_provider": "local",
            "tts_provider": "none",
            "ha_base_url": "",
            "ha_token": "",
        }
    )

    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert captured["config"]["runtime"]["background_voice_enabled"] is False

    repo_root = Path(__file__).resolve().parents[1]
    schema = json.loads((repo_root / "config" / "rex_config.schema.json").read_text())
    background_voice_schema = schema["properties"]["runtime"]["properties"][
        "background_voice_enabled"
    ]
    assert background_voice_schema["type"] == "boolean"
    assert background_voice_schema["default"] is False


def test_audio_devices_returns_sanitized_portaudio_inventory(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "rex.audio_config.list_devices",
        lambda: [
            {
                "name": "USB Microphone",
                "hostapi": 2,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
            {
                "name": "Desk Speakers",
                "hostapi": 2,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            },
        ],
    )

    rex_setup_bridge._handle_audio_devices()

    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "devices": [
            {
                "index": 0,
                "name": "USB Microphone",
                "max_input_channels": 1,
                "max_output_channels": 0,
            },
            {
                "index": 1,
                "name": "Desk Speakers",
                "max_input_channels": 0,
                "max_output_channels": 2,
            },
        ],
    }


@pytest.mark.parametrize(
    ("kind", "expected_probe"),
    [("microphone", "input"), ("speaker", "output")],
)
def test_audio_device_test_uses_non_persisting_canonical_probe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kind: str,
    expected_probe: str,
) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        "rex.audio_config.test_input_device", lambda index: calls.append(("input", index))
    )
    monkeypatch.setattr(
        "rex.audio_config.test_output_device", lambda index: calls.append(("output", index))
    )

    rex_setup_bridge._handle_test_audio_device({"kind": kind, "device_index": 3})

    assert json.loads(capsys.readouterr().out) == {"ok": True}
    assert calls == [(expected_probe, 3)]


def test_audio_device_test_rejects_invalid_kind_without_probing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "rex.audio_config.test_input_device",
        lambda _index: pytest.fail("invalid kind must not probe input"),
    )
    monkeypatch.setattr(
        "rex.audio_config.test_output_device",
        lambda _index: pytest.fail("invalid kind must not probe output"),
    )

    rex_setup_bridge._handle_test_audio_device({"kind": "camera", "device_index": 1})

    response = json.loads(capsys.readouterr().out)
    assert response["ok"] is False
    assert "microphone or speaker" in response["error"]


def test_audio_device_test_reports_probe_failure_without_persistence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_probe(_index: int) -> None:
        raise AudioDeviceError("microphone is busy")

    monkeypatch.setattr("rex.audio_config.test_input_device", fail_probe)

    rex_setup_bridge._handle_test_audio_device({"kind": "microphone", "device_index": 2})

    assert json.loads(capsys.readouterr().out) == {
        "ok": False,
        "error": "microphone is busy",
    }
