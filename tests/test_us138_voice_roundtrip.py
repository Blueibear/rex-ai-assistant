"""Tests for US-138: End-to-end voice round-trip integration test.

Acceptance criteria:
  AC1  test injects a mock wake word event, a mock STT transcript, a mock LLM
       response, and asserts TTS was called with the expected text
  AC2  test asserts the voice loop re-arms after the mock playback completes
  AC3  test passes without any real microphone, speaker, or network connection
  AC4  test added to CI and passes on first run
  AC5  Typecheck passes
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_listener(stream=None):
    """Create a WakeWordListener bypassing the heavy constructor."""
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


# Constants used across tests
FAKE_AUDIO = np.zeros(16000, dtype=np.float32)  # 1 second of silence
MOCK_TRANSCRIPT = "what is the weather today"
MOCK_LLM_RESPONSE = "It is sunny and warm today."


# ---------------------------------------------------------------------------
# AC1: inject wake word → STT → LLM → TTS pipeline
# ---------------------------------------------------------------------------


class TestVoiceRoundTrip:
    """Full pipeline exercised with mocks: wake word → STT → LLM → TTS."""

    @pytest.mark.asyncio
    async def test_tts_called_with_llm_response(self):
        """TTS (_speak_response) must be called with exactly the LLM output text."""
        listener = _make_listener()
        assistant = _make_assistant(listener=listener)

        assistant.language_model.generate = MagicMock(return_value=MOCK_LLM_RESPONSE)
        tts_calls: list[str] = []

        with (
            patch.object(assistant, "_play_wake_sound", return_value=None),
            patch.object(assistant, "_record_audio", return_value=FAKE_AUDIO),
            patch.object(
                assistant, "transcribe", new_callable=AsyncMock, return_value=MOCK_TRANSCRIPT
            ),
            patch.object(assistant, "_speak_response", side_effect=tts_calls.append),
            patch("voice_loop.append_history_entry", return_value=None),
        ):
            # Simulate a wake word event being fired by the detector
            listener._event.set()
            await assistant._handle_interaction()

        assert len(tts_calls) == 1, f"Expected 1 TTS call, got {len(tts_calls)}"
        assert (
            tts_calls[0] == MOCK_LLM_RESPONSE
        ), f"Expected TTS with {MOCK_LLM_RESPONSE!r}, got {tts_calls[0]!r}"

    @pytest.mark.asyncio
    async def test_stt_transcript_passed_to_llm(self):
        """The text returned by STT (transcribe) must be forwarded to the LLM."""
        listener = _make_listener()
        assistant = _make_assistant(listener=listener)

        llm_inputs: list[str] = []

        def mock_generate(text: str) -> str:
            llm_inputs.append(text)
            return MOCK_LLM_RESPONSE

        assistant.language_model.generate = mock_generate

        with (
            patch.object(assistant, "_play_wake_sound", return_value=None),
            patch.object(assistant, "_record_audio", return_value=FAKE_AUDIO),
            patch.object(
                assistant, "transcribe", new_callable=AsyncMock, return_value=MOCK_TRANSCRIPT
            ),
            patch.object(assistant, "_speak_response", return_value=None),
            patch("voice_loop.append_history_entry", return_value=None),
        ):
            listener._event.set()
            await assistant._handle_interaction()

        assert llm_inputs == [
            MOCK_TRANSCRIPT
        ], f"LLM must receive the STT transcript; got {llm_inputs!r}"

    @pytest.mark.asyncio
    async def test_empty_transcript_skips_llm_and_tts(self):
        """When STT returns empty string, LLM and TTS must not be called."""
        listener = _make_listener()
        assistant = _make_assistant(listener=listener)

        llm_calls: list[str] = []
        tts_calls: list[str] = []

        assistant.language_model.generate = MagicMock(
            side_effect=lambda t: llm_calls.append(t) or "response"
        )

        with (
            patch.object(assistant, "_play_wake_sound", return_value=None),
            patch.object(assistant, "_record_audio", return_value=FAKE_AUDIO),
            patch.object(assistant, "transcribe", new_callable=AsyncMock, return_value=""),
            patch.object(assistant, "_speak_response", side_effect=tts_calls.append),
            patch("voice_loop.append_history_entry", return_value=None),
        ):
            await assistant._handle_interaction()

        assert not llm_calls, "LLM must not be called when transcript is empty"
        assert not tts_calls, "TTS must not be called when transcript is empty"


# ---------------------------------------------------------------------------
# AC2: voice loop re-arms after mock playback
# ---------------------------------------------------------------------------


class TestVoiceLoopRearm:
    """Listener re-arms (event cleared, stream resumed) after interaction."""

    @pytest.mark.asyncio
    async def test_event_cleared_after_interaction(self):
        """After the interaction, the wake word event is cleared so the loop
        will block on the *next* real wake word."""
        listener = _make_listener()
        assistant = _make_assistant(listener=listener)

        assistant.language_model.generate = MagicMock(return_value=MOCK_LLM_RESPONSE)

        with (
            patch.object(assistant, "_play_wake_sound", return_value=None),
            patch.object(assistant, "_record_audio", return_value=FAKE_AUDIO),
            patch.object(
                assistant, "transcribe", new_callable=AsyncMock, return_value=MOCK_TRANSCRIPT
            ),
            patch.object(assistant, "_speak_response", return_value=None),
            patch("voice_loop.append_history_entry", return_value=None),
        ):
            # Pre-set event as if a spurious detection fired during playback
            listener._event.set()
            await assistant._handle_interaction()

        assert (
            not listener._event.is_set()
        ), "Event must be cleared after interaction so wait_for_wake blocks on next wake word"

    @pytest.mark.asyncio
    async def test_resume_stream_called_after_tts(self):
        """resume_stream() is called after TTS — confirms stream re-arm."""
        mock_stream = MagicMock()
        listener = _make_listener(stream=mock_stream)
        assistant = _make_assistant(listener=listener)

        resume_calls: list[bool] = []
        original_resume = listener.resume_stream

        def tracking_resume() -> None:
            resume_calls.append(True)
            original_resume()

        listener.resume_stream = tracking_resume

        assistant.language_model.generate = MagicMock(return_value=MOCK_LLM_RESPONSE)

        with (
            patch.object(assistant, "_play_wake_sound", return_value=None),
            patch.object(assistant, "_record_audio", return_value=FAKE_AUDIO),
            patch.object(
                assistant, "transcribe", new_callable=AsyncMock, return_value=MOCK_TRANSCRIPT
            ),
            patch.object(assistant, "_speak_response", return_value=None),
            patch("voice_loop.append_history_entry", return_value=None),
        ):
            await assistant._handle_interaction()

        assert (
            len(resume_calls) == 1
        ), "resume_stream() must be called exactly once after interaction"

    @pytest.mark.asyncio
    async def test_resume_called_even_when_tts_raises(self):
        """resume_stream() must be called even if _speak_response (TTS) raises."""
        mock_stream = MagicMock()
        listener = _make_listener(stream=mock_stream)
        assistant = _make_assistant(listener=listener)

        resume_calls: list[bool] = []
        original_resume = listener.resume_stream

        def tracking_resume() -> None:
            resume_calls.append(True)
            original_resume()

        listener.resume_stream = tracking_resume

        assistant.language_model.generate = MagicMock(return_value=MOCK_LLM_RESPONSE)

        with (
            patch.object(assistant, "_play_wake_sound", return_value=None),
            patch.object(assistant, "_record_audio", return_value=FAKE_AUDIO),
            patch.object(
                assistant, "transcribe", new_callable=AsyncMock, return_value=MOCK_TRANSCRIPT
            ),
            patch.object(
                assistant,
                "_speak_response",
                side_effect=Exception("simulated TTS failure"),
            ),
            patch("voice_loop.append_history_entry", return_value=None),
        ):
            await assistant._handle_interaction()  # must not propagate

        assert resume_calls, "resume_stream() must be called even when TTS raises"


# ---------------------------------------------------------------------------
# AC3: no real hardware or network required
# ---------------------------------------------------------------------------


class TestNoHardwareRequired:
    """Confirms the entire test suite runs without mic, speaker, or network."""

    @pytest.mark.asyncio
    async def test_full_pipeline_no_real_devices(self):
        """Complete wake→STT→LLM→TTS cycle uses only mocks — no hardware."""
        mock_stream = MagicMock()
        listener = _make_listener(stream=mock_stream)
        assistant = _make_assistant(listener=listener)

        assistant.language_model.generate = MagicMock(return_value="Hello from Rex")

        tts_received: list[str] = []

        with (
            patch.object(assistant, "_play_wake_sound", return_value=None),
            patch.object(assistant, "_record_audio", return_value=np.zeros(8000, dtype=np.float32)),
            patch.object(assistant, "transcribe", new_callable=AsyncMock, return_value="hello rex"),
            patch.object(assistant, "_speak_response", side_effect=tts_received.append),
            patch("voice_loop.append_history_entry", return_value=None),
        ):
            await assistant._handle_interaction()

        assert tts_received == ["Hello from Rex"]

    @pytest.mark.asyncio
    async def test_wait_for_wake_resolved_by_injected_event(self):
        """wait_for_wake() resolves immediately when event is pre-set in test."""
        listener = _make_listener()
        # Simulate wake word detection by setting the event directly
        listener._event.set()

        await listener.wait_for_wake()

        # After wait_for_wake returns the event is cleared
        assert not listener._event.is_set(), "wait_for_wake must clear the event after returning"

    @pytest.mark.asyncio
    async def test_two_consecutive_interactions_both_succeed(self):
        """Simulate two wake words in sequence; both produce TTS calls."""
        listener = _make_listener(stream=MagicMock())
        assistant = _make_assistant(listener=listener)

        assistant.language_model.generate = MagicMock(return_value="reply")
        tts_calls: list[str] = []

        with (
            patch.object(assistant, "_play_wake_sound", return_value=None),
            patch.object(assistant, "_record_audio", return_value=FAKE_AUDIO),
            patch.object(
                assistant, "transcribe", new_callable=AsyncMock, return_value=MOCK_TRANSCRIPT
            ),
            patch.object(assistant, "_speak_response", side_effect=tts_calls.append),
            patch("voice_loop.append_history_entry", return_value=None),
        ):
            # First wake word interaction
            await assistant._handle_interaction()
            # Second wake word interaction — listener must have re-armed
            await assistant._handle_interaction()

        assert len(tts_calls) == 2, f"Both interactions must produce TTS; got {len(tts_calls)}"


# ---------------------------------------------------------------------------
# AC5: Structural / typecheck helpers
# ---------------------------------------------------------------------------


class TestStructural:
    def test_async_rex_assistant_importable(self):
        from voice_loop import AsyncRexAssistant

        assert AsyncRexAssistant is not None

    def test_wake_word_listener_importable(self):
        from voice_loop import WakeWordListener

        assert WakeWordListener is not None

    def test_process_conversation_is_coroutine(self):
        from voice_loop import AsyncRexAssistant

        assert inspect.iscoroutinefunction(AsyncRexAssistant._process_conversation)

    def test_handle_interaction_is_coroutine(self):
        from voice_loop import AsyncRexAssistant

        assert inspect.iscoroutinefunction(AsyncRexAssistant._handle_interaction)

    def test_wait_for_wake_is_coroutine(self):
        from voice_loop import WakeWordListener

        assert inspect.iscoroutinefunction(WakeWordListener.wait_for_wake)

    def test_speak_response_exists(self):
        from voice_loop import AsyncRexAssistant

        assert hasattr(
            AsyncRexAssistant, "_speak_response"
        ), "AsyncRexAssistant must have _speak_response"

    def test_transcribe_is_coroutine(self):
        from voice_loop import AsyncRexAssistant

        assert inspect.iscoroutinefunction(AsyncRexAssistant.transcribe)
