"""Tests for US-P6-009: USE_OPENCLAW_VOICE_BACKEND flag in rex/voice_loop.py.

Current contract (US-097):
  - The canonical voice loop always keeps the passed-in Assistant brain.
  - The legacy flag never swaps in VoiceBridge.
  - OpenClaw remains available as an external capability/compatibility provider.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _make_mock_client() -> MagicMock:
    """Return a mock OpenClaw client whose /health check succeeds."""
    mock_client = MagicMock()
    mock_client.get.return_value = {"status": "ok"}
    return mock_client


def _make_voice_loop(settings_override: dict, mock_assistant: MagicMock) -> object:
    """Construct a VoiceLoop with all heavy deps stubbed and settings patched."""
    use_openclaw = settings_override.get("use_openclaw_voice_backend", False)
    mock_client = _make_mock_client() if use_openclaw else None

    with patch("rex.voice_loop.settings") as mock_settings:
        for attr, value in settings_override.items():
            setattr(mock_settings, attr, value)
        mock_settings.openclaw_gateway_url = "http://localhost:8765"
        # Ensure attribute access falls back sensibly for unset attrs
        mock_settings.__class__ = type("_Cfg", (), {})

        with patch("rex.openclaw.http_client.get_openclaw_client", return_value=mock_client):
            from rex.voice_loop import VoiceLoop

            return VoiceLoop(
                mock_assistant,
                wake_listener=MagicMock(),
                detection_source=MagicMock(),
                record_phrase=MagicMock(),
                transcribe=MagicMock(),
                speak=MagicMock(),
            )


class TestVoiceLoopFlagOff:
    """When use_openclaw_voice_backend=False, VoiceLoop keeps the passed-in assistant."""

    def test_flag_false_keeps_original_assistant(self):
        """_assistant is the passed-in object when flag is False."""
        mock_assistant = MagicMock()
        vl = _make_voice_loop({"use_openclaw_voice_backend": False}, mock_assistant)
        assert vl._assistant is mock_assistant

    def test_flag_false_does_not_instantiate_voice_bridge(self):
        """VoiceBridge() is never called when flag is False."""
        mock_assistant = MagicMock()
        with patch("rex.openclaw.voice_bridge.VoiceBridge") as mock_cls:
            vl = _make_voice_loop({"use_openclaw_voice_backend": False}, mock_assistant)
            mock_cls.assert_not_called()
            assert vl._assistant is mock_assistant

    def test_flag_absent_keeps_original_assistant(self):
        """Missing attribute behaves like False (getattr default)."""
        mock_assistant = MagicMock()
        with patch("rex.voice_loop.settings") as mock_settings:
            # Simulate attribute not present
            del mock_settings.use_openclaw_voice_backend

            from rex.voice_loop import VoiceLoop

            vl = VoiceLoop(
                mock_assistant,
                wake_listener=MagicMock(),
                detection_source=MagicMock(),
                record_phrase=MagicMock(),
                transcribe=MagicMock(),
                speak=MagicMock(),
            )
        assert vl._assistant is mock_assistant


class TestVoiceLoopFlagOn:
    """Legacy flag cannot replace the canonical TurnEngine-backed Assistant."""

    def test_flag_true_keeps_original_assistant(self):
        mock_assistant = MagicMock()
        vl = _make_voice_loop({"use_openclaw_voice_backend": True}, mock_assistant)
        assert vl._assistant is mock_assistant

    def test_flag_true_does_not_instantiate_voice_bridge(self):
        mock_assistant = MagicMock()
        with patch("rex.openclaw.voice_bridge.VoiceBridge") as mock_cls:
            vl = _make_voice_loop({"use_openclaw_voice_backend": True}, mock_assistant)
        mock_cls.assert_not_called()
        assert vl._assistant is mock_assistant
