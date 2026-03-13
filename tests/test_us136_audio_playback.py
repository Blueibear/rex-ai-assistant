"""Tests for US-136: Fix audio playback and output device selection."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest


def _make_assistant(output_device=None):
    """Create a minimal AsyncRexAssistant with mocked dependencies."""
    from voice_loop import AsyncRexAssistant

    assistant = AsyncRexAssistant.__new__(AsyncRexAssistant)
    assistant.config = MagicMock()
    assistant.config.audio_output_device = output_device
    assistant.config.tts_speed = 1.0
    assistant.config.conversation_export = False
    assistant.user_voice_refs = {}
    assistant.active_user = "test_user"
    assistant._tts = None
    assistant._whisper_model = None
    assistant._sample_rate = 16000
    return assistant


def _make_fake_tts_to_file(wav_bytes: bytes):
    """Return a tts_to_file side_effect that writes a minimal WAV."""
    import io
    import struct
    import wave

    def _side_effect(text, file_path, **kwargs):
        # Write a valid 1-sample WAV to the given path
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(22050)
            w.writeframes(struct.pack("<h", 0))
        Path(file_path).write_bytes(buf.getvalue())

    from pathlib import Path

    return _side_effect


# ---------------------------------------------------------------------------
# AC1 + AC2: sounddevice used for playback with correct device
# ---------------------------------------------------------------------------


class TestSounddevicePlayback:
    """sounddevice is used for playback and honours audio_output_device."""

    def _run_speak(self, assistant, mock_sd, mock_sf, mock_tts_class):
        """Wire mocks and call _speak_response synchronously."""
        from pathlib import Path

        import numpy as np

        # Build a real WAV in memory for soundfile.read to return
        dummy_audio = np.zeros(22050, dtype="float32")
        dummy_rate = 22050

        mock_sf.read.return_value = (dummy_audio, dummy_rate)

        mock_tts = MagicMock()

        def _tts_to_file(text, file_path, **kwargs):
            Path(file_path).write_bytes(b"")

        mock_tts.tts_to_file.side_effect = _tts_to_file
        assistant._tts = mock_tts

        with (
            patch("voice_loop._load_sounddevice", return_value=mock_sd),
            patch("voice_loop._lazy_import_soundfile", return_value=mock_sf),
            patch("voice_loop.chunk_text_for_xtts", return_value=["hello world"]),
        ):
            assistant._speak_response("hello world")

    def test_sounddevice_play_called_with_configured_device(self):
        """sounddevice.play is called with the configured output device index."""

        assistant = _make_assistant(output_device=3)
        mock_sd = MagicMock()
        mock_sf = MagicMock()

        self._run_speak(assistant, mock_sd, mock_sf, None)

        mock_sd.play.assert_called_once()
        _, kwargs = mock_sd.play.call_args
        assert kwargs.get("device") == 3

    def test_sounddevice_play_called_with_none_for_default_device(self):
        """sounddevice.play uses device=None (system default) when not configured."""

        assistant = _make_assistant(output_device=None)
        mock_sd = MagicMock()
        mock_sf = MagicMock()

        self._run_speak(assistant, mock_sd, mock_sf, None)

        mock_sd.play.assert_called_once()
        _, kwargs = mock_sd.play.call_args
        assert kwargs.get("device") is None

    def test_sounddevice_play_called_with_string_device_name(self):
        """sounddevice.play accepts a string device name from config."""

        assistant = _make_assistant(output_device="Speakers (High Definition Audio)")
        mock_sd = MagicMock()
        mock_sf = MagicMock()

        self._run_speak(assistant, mock_sd, mock_sf, None)

        mock_sd.play.assert_called_once()
        _, kwargs = mock_sd.play.call_args
        assert kwargs.get("device") == "Speakers (High Definition Audio)"

    def test_sounddevice_blocking_true(self):
        """sounddevice.play is called with blocking=True so it waits for completion."""
        assistant = _make_assistant(output_device=None)
        mock_sd = MagicMock()
        mock_sf = MagicMock()

        self._run_speak(assistant, mock_sd, mock_sf, None)

        _, kwargs = mock_sd.play.call_args
        assert kwargs.get("blocking") is True

    def test_sounddevice_stop_called_after_play(self):
        """sounddevice.stop() is called after play to release the device."""
        assistant = _make_assistant(output_device=None)
        mock_sd = MagicMock()
        mock_sf = MagicMock()

        self._run_speak(assistant, mock_sd, mock_sf, None)

        mock_sd.stop.assert_called()


# ---------------------------------------------------------------------------
# AC3: playback does not block the voice loop after audio ends
# ---------------------------------------------------------------------------


class TestNonBlockingVoiceLoop:
    """After _speak_response returns, _process_conversation completes normally."""

    def test_speak_response_is_synchronous_method(self):
        """_speak_response is a regular function, not a coroutine.

        This ensures it can be safely dispatched via asyncio.to_thread,
        which is the pattern that keeps the event loop unblocked during playback.
        """
        import inspect

        from voice_loop import AsyncRexAssistant

        assert not inspect.iscoroutinefunction(
            AsyncRexAssistant._speak_response
        ), "_speak_response must be a sync method so asyncio.to_thread can run it in a thread"

    def test_process_conversation_dispatches_speak_via_to_thread(self):
        """_process_conversation calls _speak_response via asyncio.to_thread.

        This is the mechanism that prevents playback from blocking the event loop.
        """
        import inspect

        from voice_loop import AsyncRexAssistant

        src = inspect.getsource(AsyncRexAssistant._process_conversation)
        assert "asyncio.to_thread" in src, "_process_conversation must use asyncio.to_thread"
        assert "_speak_response" in src, "_speak_response must be called in _process_conversation"


# ---------------------------------------------------------------------------
# AC4: audio playback errors are caught and logged
# ---------------------------------------------------------------------------


class TestPlaybackErrorHandling:
    """Playback errors are explicitly logged and not silently swallowed."""

    def test_playback_error_is_logged_with_exc_info(self, caplog):
        """When sounddevice.play raises, the error is logged via logger.error."""
        import numpy as np

        from rex.assistant_errors import TextToSpeechError

        assistant = _make_assistant(output_device=None)
        mock_sd = MagicMock()
        mock_sd.play.side_effect = RuntimeError("device busy")
        mock_sf = MagicMock()
        mock_sf.read.return_value = (np.zeros(22050, dtype="float32"), 22050)

        mock_tts = MagicMock()
        from pathlib import Path

        def _tts_to_file(text, file_path, **kwargs):
            Path(file_path).write_bytes(b"")

        mock_tts.tts_to_file.side_effect = _tts_to_file
        assistant._tts = mock_tts

        with (
            patch("voice_loop._load_sounddevice", return_value=mock_sd),
            patch("voice_loop._lazy_import_soundfile", return_value=mock_sf),
            patch("voice_loop.chunk_text_for_xtts", return_value=["hello"]),
            caplog.at_level(logging.ERROR, logger="voice_loop"),
        ):
            with pytest.raises(TextToSpeechError, match="Audio playback failed"):
                assistant._speak_response("hello")

        assert any(
            "Audio playback failed" in r.message for r in caplog.records
        ), "Expected logger.error call for playback failure"

    def test_playback_error_raises_tts_error_not_silent(self):
        """Playback errors propagate as TextToSpeechError, not swallowed."""
        import numpy as np

        from rex.assistant_errors import TextToSpeechError

        assistant = _make_assistant(output_device=None)
        mock_sd = MagicMock()
        mock_sd.play.side_effect = OSError("no such device")
        mock_sf = MagicMock()
        mock_sf.read.return_value = (np.zeros(22050, dtype="float32"), 22050)

        mock_tts = MagicMock()
        from pathlib import Path

        def _tts_to_file(text, file_path, **kwargs):
            Path(file_path).write_bytes(b"")

        mock_tts.tts_to_file.side_effect = _tts_to_file
        assistant._tts = mock_tts

        with (
            patch("voice_loop._load_sounddevice", return_value=mock_sd),
            patch("voice_loop._lazy_import_soundfile", return_value=mock_sf),
            patch("voice_loop.chunk_text_for_xtts", return_value=["hello"]),
        ):
            with pytest.raises(TextToSpeechError):
                assistant._speak_response("hello")

    def test_synthesis_error_distinct_from_playback_error(self):
        """Synthesis failures produce 'Failed to synthesise speech' message."""
        from rex.assistant_errors import TextToSpeechError

        assistant = _make_assistant(output_device=None)
        mock_tts = MagicMock()
        mock_tts.tts_to_file.side_effect = RuntimeError("model crashed")
        assistant._tts = mock_tts

        mock_sf = MagicMock()
        mock_sd = MagicMock()

        with (
            patch("voice_loop._load_sounddevice", return_value=mock_sd),
            patch("voice_loop._lazy_import_soundfile", return_value=mock_sf),
            patch("voice_loop.chunk_text_for_xtts", return_value=["hello"]),
        ):
            with pytest.raises(TextToSpeechError, match="Failed to synthesise speech"):
                assistant._speak_response("hello")

        # sounddevice.play should NOT have been called
        mock_sd.play.assert_not_called()
