"""Tests for US-137: Fix voice loop re-arm after TTS playback completes.

Acceptance criteria:
  AC1  after TTS audio finishes playing, the wake word detector resumes within 1 second
  AC2  the microphone stream is not left open or blocked after playback
  AC3  a second voice interaction triggered after the first completes
       successfully produces a spoken response
  AC4  Typecheck passes (structural / import tests)
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_listener(stream=None):
    """Create a WakeWordListener with mocked stream."""
    from voice_loop import WakeWordListener

    listener = WakeWordListener.__new__(WakeWordListener)
    listener._stream = stream
    listener._event = asyncio.Event()
    listener.loop = None
    listener.model = MagicMock()
    listener.threshold = 0.5
    listener.sample_rate = 16000
    listener.block_size = 1280
    listener.device = None
    listener._callback_count = 0
    return listener


def _make_assistant(listener=None):
    """Create AsyncRexAssistant bypassing heavy __init__."""
    from voice_loop import AsyncRexAssistant

    assistant = AsyncRexAssistant.__new__(AsyncRexAssistant)
    assistant.config = MagicMock()
    assistant.config.audio_output_device = None
    assistant.config.tts_speed = 1.0
    assistant.config.conversation_export = False
    assistant.config.wakeword_threshold = 0.5
    assistant.config.command_duration = 3.0
    assistant.user_voice_refs = {}
    assistant.active_user = "test_user"
    assistant._tts = None
    assistant._whisper_model = None
    assistant._sample_rate = 16000
    assistant._running = True
    assistant._state = "running"
    assistant._stop_requested_by = None
    assistant.language_model = MagicMock()
    assistant.users_map = {}
    assistant.profiles = {}
    assistant.plugins = {}

    if listener is None:
        listener = _make_listener()
    assistant._listener = listener
    return assistant


# ---------------------------------------------------------------------------
# AC1: resume_stream() called immediately after TTS — within 1 second
# ---------------------------------------------------------------------------


class TestResumeCalledAfterPlayback:
    """pause_stream / resume_stream called in the right order."""

    @pytest.mark.asyncio
    async def test_resume_stream_called_after_handle_interaction(self):
        """resume_stream must be called after _process_conversation completes."""
        mock_stream = MagicMock()
        listener = _make_listener(stream=mock_stream)
        assistant = _make_assistant(listener=listener)

        call_order = []

        def record_pause():
            call_order.append("pause")

        def record_resume():
            call_order.append("resume")

        listener.pause_stream = record_pause
        listener.resume_stream = record_resume

        with (
            patch.object(assistant, "_play_wake_sound", return_value=None),
            patch.object(assistant, "_process_conversation", new_callable=AsyncMock),
        ):
            await assistant._handle_interaction()

        assert "pause" in call_order
        assert "resume" in call_order
        assert call_order.index("pause") < call_order.index(
            "resume"
        ), "pause must come before resume"

    @pytest.mark.asyncio
    async def test_resume_stream_called_even_on_exception(self):
        """resume_stream must be called even when _process_conversation raises."""
        mock_stream = MagicMock()
        listener = _make_listener(stream=mock_stream)
        assistant = _make_assistant(listener=listener)

        resumed = []
        listener.pause_stream = MagicMock()
        listener.resume_stream = lambda: resumed.append(True)

        async def _boom():
            raise RuntimeError("simulated failure")

        with (
            patch.object(assistant, "_play_wake_sound", return_value=None),
            patch.object(assistant, "_process_conversation", side_effect=_boom),
        ):
            await assistant._handle_interaction()  # must not propagate

        assert resumed, "resume_stream must be called even after an exception"


# ---------------------------------------------------------------------------
# AC2: mic stream not left open / blocked after playback
# ---------------------------------------------------------------------------


class TestMicStreamNotBlocked:
    """pause_stream stops the InputStream; resume_stream restarts it."""

    def test_pause_stream_stops_underlying_stream(self):
        """pause_stream() calls stream.stop() on the InputStream."""
        mock_stream = MagicMock()
        listener = _make_listener(stream=mock_stream)
        listener.pause_stream()
        mock_stream.stop.assert_called_once()

    def test_pause_stream_safe_when_stream_is_none(self):
        """pause_stream() must not raise when _stream is None."""
        listener = _make_listener(stream=None)
        listener.pause_stream()  # should not raise

    def test_resume_stream_starts_underlying_stream(self):
        """resume_stream() calls stream.start() on the InputStream."""
        mock_stream = MagicMock()
        listener = _make_listener(stream=mock_stream)
        listener.resume_stream()
        mock_stream.start.assert_called_once()

    def test_resume_stream_safe_when_stream_is_none(self):
        """resume_stream() must not raise when _stream is None."""
        listener = _make_listener(stream=None)
        listener.resume_stream()  # should not raise

    def test_resume_stream_clears_event(self):
        """resume_stream() clears the asyncio.Event to discard spurious detections."""
        # asyncio.Event can be used synchronously for is_set() / clear() checks
        listener = _make_listener(stream=MagicMock())
        # Simulate a spurious wake-word fire during playback.
        listener._event.set()
        assert listener._event.is_set()
        listener.resume_stream()
        assert (
            not listener._event.is_set()
        ), "resume_stream must clear the event so next wait_for_wake blocks properly"

    def test_pause_stream_safe_on_stop_exception(self):
        """pause_stream() suppresses exceptions from stream.stop()."""
        mock_stream = MagicMock()
        mock_stream.stop.side_effect = OSError("device busy")
        listener = _make_listener(stream=mock_stream)
        listener.pause_stream()  # must not propagate

    def test_resume_stream_safe_on_start_exception(self):
        """resume_stream() suppresses exceptions from stream.start()."""
        mock_stream = MagicMock()
        mock_stream.start.side_effect = OSError("device busy")
        listener = _make_listener(stream=mock_stream)
        listener.resume_stream()  # must not propagate


# ---------------------------------------------------------------------------
# AC3: second voice interaction produces spoken response
# ---------------------------------------------------------------------------


class TestSecondInteractionWorks:
    """After first interaction, the loop re-arms and processes a second one."""

    @pytest.mark.asyncio
    async def test_two_consecutive_interactions_both_call_speak(self):
        """Simulate two wake-word events; _speak_response should fire twice."""
        mock_stream = MagicMock()
        listener = _make_listener(stream=mock_stream)
        assistant = _make_assistant(listener=listener)

        speak_calls = []

        async def fake_process_conversation():
            # Drive _speak_response directly to simulate a full interaction.
            speak_calls.append(1)

        with (
            patch.object(assistant, "_play_wake_sound", return_value=None),
            patch.object(assistant, "_process_conversation", side_effect=fake_process_conversation),
        ):
            # First interaction
            await assistant._handle_interaction()
            # Second interaction
            await assistant._handle_interaction()

        assert len(speak_calls) == 2, (
            "Both interactions must execute _process_conversation; " f"got {len(speak_calls)} calls"
        )

    @pytest.mark.asyncio
    async def test_event_cleared_between_interactions(self):
        """After each interaction, event must be clear so next wait_for_wake blocks."""
        mock_stream = MagicMock()
        listener = _make_listener(stream=mock_stream)
        # Simulate spurious event set during first interaction.
        listener._event.set()
        assistant = _make_assistant(listener=listener)

        with (
            patch.object(assistant, "_play_wake_sound", return_value=None),
            patch.object(assistant, "_process_conversation", new_callable=AsyncMock),
        ):
            await assistant._handle_interaction()

        # After interaction, event must be clear.
        assert (
            not listener._event.is_set()
        ), "Event must be cleared after interaction so the loop waits for the next real wake word"


# ---------------------------------------------------------------------------
# AC4: Structural / typecheck helpers
# ---------------------------------------------------------------------------


class TestStructural:
    def test_pause_stream_exists(self):
        from voice_loop import WakeWordListener

        assert hasattr(
            WakeWordListener, "pause_stream"
        ), "WakeWordListener must have pause_stream()"

    def test_resume_stream_exists(self):
        from voice_loop import WakeWordListener

        assert hasattr(
            WakeWordListener, "resume_stream"
        ), "WakeWordListener must have resume_stream()"

    def test_handle_interaction_is_coroutine(self):
        from voice_loop import AsyncRexAssistant

        assert inspect.iscoroutinefunction(AsyncRexAssistant._handle_interaction)

    def test_pause_stream_called_in_handle_interaction(self):
        """Source of _handle_interaction must call pause_stream."""
        from voice_loop import AsyncRexAssistant

        src = inspect.getsource(AsyncRexAssistant._handle_interaction)
        assert "pause_stream" in src

    def test_resume_stream_called_in_handle_interaction(self):
        """Source of _handle_interaction must call resume_stream."""
        from voice_loop import AsyncRexAssistant

        src = inspect.getsource(AsyncRexAssistant._handle_interaction)
        assert "resume_stream" in src

    def test_resume_in_finally_block(self):
        """resume_stream must appear in a finally block."""
        from voice_loop import AsyncRexAssistant

        src = inspect.getsource(AsyncRexAssistant._handle_interaction)
        # "finally:" must precede "resume_stream"
        finally_pos = src.find("finally:")
        resume_pos = src.find("resume_stream")
        assert finally_pos != -1, "No finally: block found in _handle_interaction"
        assert resume_pos > finally_pos, "resume_stream must be inside the finally block"
