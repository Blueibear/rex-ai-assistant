"""Tests for US-311: Fix Settings > Audio Output page."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_SCRIPT = REPO_ROOT / "rex_speaker_bridge.py"


# ---------------------------------------------------------------------------
# AC: bridge script exists and is importable
# ---------------------------------------------------------------------------


def test_speaker_bridge_script_exists() -> None:
    """rex_speaker_bridge.py must exist at repo root."""
    assert BRIDGE_SCRIPT.exists(), f"Bridge script missing: {BRIDGE_SCRIPT}"


def test_speaker_discovery_importable() -> None:
    """SpeakerDiscoveryService must be importable from rex.audio.speaker_discovery."""
    from rex.audio.speaker_discovery import SpeakerDiscoveryService  # noqa: F401

    svc = SpeakerDiscoveryService(
        refresh_interval_seconds=60.0, discovery_timeout_seconds=0.1
    )
    assert svc is not None


# ---------------------------------------------------------------------------
# AC: bridge called via centralized resolver returns valid structure
# ---------------------------------------------------------------------------


def _call_bridge_main(stdin_text: str) -> dict:
    """Import and call rex_speaker_bridge.main() with mocked stdin/discovery."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("rex_speaker_bridge", BRIDGE_SCRIPT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    captured: list[str] = []

    with (
        patch.object(sys, "stdin", io.StringIO(stdin_text)),
        patch("builtins.print", side_effect=lambda *args, **_kw: captured.append(str(args[0]))),
    ):
        mod.main()

    json_lines = [line for line in captured if line.strip().startswith("{")]
    assert json_lines, f"No JSON output from bridge. Captured: {captured}"
    return json.loads(json_lines[-1])


def test_list_command_returns_ok_and_speakers_key() -> None:
    """{command: list} must return {ok: true, speakers: [...]}."""
    from rex.audio.speaker_discovery import DiscoveredSpeaker

    fake_speakers = [
        DiscoveredSpeaker(provider="sonos", name="Living Room", ip="192.168.1.50", model="Play:1")
    ]

    with patch("rex.audio.speaker_discovery.SpeakerDiscoveryService") as mock_cls:
        mock_svc = MagicMock()
        mock_svc.discover_now.return_value = fake_speakers
        mock_cls.return_value = mock_svc
        result = _call_bridge_main('{"command": "list"}')

    assert result.get("ok") is True
    assert "speakers" in result
    speakers = result["speakers"]
    assert len(speakers) == 1
    assert speakers[0]["provider"] == "sonos"
    assert speakers[0]["name"] == "Living Room"
    assert speakers[0]["ip"] == "192.168.1.50"


def test_list_command_empty_payload_defaults_to_list() -> None:
    """Empty payload should default command to 'list'."""
    with patch("rex.audio.speaker_discovery.SpeakerDiscoveryService") as mock_cls:
        mock_svc = MagicMock()
        mock_svc.discover_now.return_value = []
        mock_cls.return_value = mock_svc
        result = _call_bridge_main("{}")

    assert result.get("ok") is True
    assert result.get("speakers") == []


def test_unknown_command_returns_error() -> None:
    """Unknown command must return {ok: false, error: ...}."""
    result = _call_bridge_main('{"command": "reboot"}')
    assert result.get("ok") is False
    assert "error" in result


def test_bad_json_input_returns_error() -> None:
    """Malformed JSON input must return an error, not crash."""
    result = _call_bridge_main("not valid json at all")
    assert result.get("ok") is False
    assert "error" in result


def test_discovery_exception_returns_error_not_crash() -> None:
    """If SpeakerDiscoveryService.discover_now raises, bridge returns an error."""
    with patch("rex.audio.speaker_discovery.SpeakerDiscoveryService") as mock_cls:
        mock_svc = MagicMock()
        mock_svc.discover_now.side_effect = RuntimeError("network unreachable")
        mock_cls.return_value = mock_svc
        result = _call_bridge_main('{"command": "list"}')

    assert result.get("ok") is False
    assert "speakers" in result
    assert result["speakers"] == []


# ---------------------------------------------------------------------------
# AC: bridge uses centralized resolver (code inspection)
# ---------------------------------------------------------------------------


def test_bridge_uses_centralized_resolver() -> None:
    """Bridge must import resolveBridgePath / resolve_python from bridge_utils."""
    source = BRIDGE_SCRIPT.read_text(encoding="utf-8")
    assert "bridge_utils" in source, (
        "Bridge must import from rex.bridge_utils to use the centralized resolver pattern"
    )


# ---------------------------------------------------------------------------
# AC: subprocess invocation succeeds (real network call, no crashes)
# ---------------------------------------------------------------------------


def test_bridge_subprocess_returns_json() -> None:
    """Running the bridge via subprocess with list command must return valid JSON."""
    result = subprocess.run(
        [sys.executable, str(BRIDGE_SCRIPT)],
        input='{"command": "list"}',
        capture_output=True,
        text=True,
        timeout=15,
    )
    # Bridge should exit 0 even when no devices are found
    json_lines = [
        line
        for line in result.stdout.strip().splitlines()
        if line.strip().startswith("{")
    ]
    assert json_lines, (
        f"Expected JSON on stdout.\nstdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
    )
    data = json.loads(json_lines[-1])
    assert "speakers" in data, f"Missing 'speakers' key in response: {data}"
