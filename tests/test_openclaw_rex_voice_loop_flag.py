"""Tests for US-P6-009: USE_OPENCLAW_VOICE_BACKEND flag in rex/voice_loop.py.

Acceptance criteria:
  - When use_openclaw_voice_backend=False, _assistant keeps the passed-in assistant
  - When use_openclaw_voice_backend=True, _assistant is replaced with VoiceBridge()
  - VoiceBridge creation failure falls back gracefully (keeps previous _assistant)
  - Tests pass
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_voice_loop(settings_override: dict, mock_assistant: MagicMock) -> object:
    """Construct a VoiceLoop with all heavy deps stubbed and settings patched."""
    with patch("rex.voice_loop.settings") as mock_settings:
        for attr, value in settings_override.items():
            setattr(mock_settings, attr, value)
        # Ensure attribute access falls back sensibly for unset attrs
        mock_settings.__class__ = type("_Cfg", (), {})

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
    """When use_openclaw_voice_backend=True, _assistant is replaced with VoiceBridge."""

    def test_flag_true_replaces_assistant_with_voice_bridge(self):
        """_assistant is a VoiceBridge instance when flag is True."""
        from rex.openclaw.voice_bridge import VoiceBridge

        mock_assistant = MagicMock()
        mock_bridge = MagicMock(spec=VoiceBridge)

        with patch("rex.openclaw.voice_bridge.VoiceBridge", return_value=mock_bridge):
            vl = _make_voice_loop({"use_openclaw_voice_backend": True}, mock_assistant)

        assert vl._assistant is mock_bridge

    def test_flag_true_voice_bridge_instantiated_once(self):
        """VoiceBridge() is called exactly once when flag is True."""
        from rex.openclaw.voice_bridge import VoiceBridge

        mock_assistant = MagicMock()

        with patch(
            "rex.openclaw.voice_bridge.VoiceBridge", return_value=MagicMock(spec=VoiceBridge)
        ) as mock_cls:
            _make_voice_loop({"use_openclaw_voice_backend": True}, mock_assistant)

        mock_cls.assert_called_once()

    def test_flag_true_voice_bridge_failure_falls_back(self):
        """If VoiceBridge() raises, _assistant keeps the original passed-in value."""
        mock_assistant = MagicMock()

        with patch(
            "rex.openclaw.voice_bridge.VoiceBridge",
            side_effect=RuntimeError("bridge unavailable"),
        ):
            vl = _make_voice_loop({"use_openclaw_voice_backend": True}, mock_assistant)

        assert vl._assistant is mock_assistant
