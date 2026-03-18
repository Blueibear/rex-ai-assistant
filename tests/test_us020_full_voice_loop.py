"""US-020: Full voice interaction loop tests.

Acceptance criteria:
- wake word triggers listening
- STT produces transcript
- LLM response generated
- response spoken aloud
- Typecheck passes
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from rex.voice_loop import VoiceLoop

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


def _make_wake_listener(trigger_count: int = 1):
    """Return a mock wake listener that yields *trigger_count* times."""

    async def _listen(_detection_source):
        for _ in range(trigger_count):
            yield True

    listener = MagicMock()
    listener.listen = _listen
    return listener


def _make_voice_loop(
    *,
    wake_listener=None,
    audio=None,
    transcript: str = "hello rex",
    llm_response: str = "Hello there",
    speak_spy=None,
    acknowledge_spy=None,
):
    """Build a VoiceLoop wired with async mock callables."""
    if audio is None:
        audio = np.zeros(16000, dtype="float32")
    if wake_listener is None:
        wake_listener = _make_wake_listener(trigger_count=1)

    detection_source = AsyncMock(return_value=audio)
    record_phrase = AsyncMock(return_value=audio)
    transcribe = AsyncMock(return_value=transcript)

    assistant = MagicMock()
    assistant.generate_reply = AsyncMock(return_value=llm_response)

    speak = speak_spy or AsyncMock()
    acknowledge = acknowledge_spy

    loop = VoiceLoop(
        assistant,
        wake_listener=wake_listener,
        detection_source=detection_source,
        record_phrase=record_phrase,
        transcribe=transcribe,
        speak=speak,
        acknowledge=acknowledge,
    )
    return loop, assistant, transcribe, speak, record_phrase


# ---------------------------------------------------------------------------
# 1. Wake word triggers listening
# ---------------------------------------------------------------------------


class TestWakeWordTriggersListening:
    def test_run_executes_on_wake_word(self):
        """VoiceLoop.run() proceeds past wake word and attempts STT."""
        loop, _, transcribe, speak, _ = _make_voice_loop()
        asyncio.run(loop.run(max_interactions=1))
        transcribe.assert_awaited_once()

    def test_no_iterations_when_wake_word_never_fires(self):
        """When the wake listener never yields, speak is never called."""
        wake_listener = _make_wake_listener(trigger_count=0)
        loop, _, transcribe, speak, _ = _make_voice_loop(wake_listener=wake_listener)
        asyncio.run(loop.run())
        speak.assert_not_awaited()
        transcribe.assert_not_awaited()

    def test_multiple_wake_words_produce_multiple_interactions(self):
        """Three wake word events produce three full interaction cycles."""
        wake_listener = _make_wake_listener(trigger_count=3)
        loop, assistant, transcribe, speak, _ = _make_voice_loop(wake_listener=wake_listener)
        asyncio.run(loop.run())
        assert transcribe.await_count == 3
        assert speak.await_count == 3

    def test_max_interactions_limits_wake_word_processing(self):
        """max_interactions=1 stops after one cycle even if listener yields more."""
        wake_listener = _make_wake_listener(trigger_count=5)
        loop, _, _, speak, _ = _make_voice_loop(wake_listener=wake_listener)
        asyncio.run(loop.run(max_interactions=1))
        assert speak.await_count == 1

    def test_acknowledge_called_after_wake_word(self):
        """acknowledge() is called once the wake word fires."""
        ack = AsyncMock()
        loop, _, _, _, _ = _make_voice_loop(acknowledge_spy=ack)
        asyncio.run(loop.run(max_interactions=1))
        ack.assert_awaited_once()


# ---------------------------------------------------------------------------
# 2. STT produces transcript
# ---------------------------------------------------------------------------


class TestSTTProducesTranscript:
    def test_transcribe_receives_recorded_audio(self):
        """transcribe() is called with the audio returned by record_phrase()."""
        audio = np.ones(16000, dtype="float32")
        loop, _, transcribe, _, _ = _make_voice_loop(audio=audio)
        asyncio.run(loop.run(max_interactions=1))
        transcribe.assert_awaited_once_with(audio)

    def test_empty_transcript_skips_llm(self):
        """When transcribe() returns empty string, LLM is not called."""
        loop, assistant, _, _, _ = _make_voice_loop(transcript="")
        asyncio.run(loop.run(max_interactions=1))
        assistant.generate_reply.assert_not_awaited()

    def test_stt_error_continues_loop(self):
        """STT errors are caught and the loop continues without crashing."""
        from rex.assistant_errors import SpeechToTextError

        wake_listener = _make_wake_listener(trigger_count=2)
        audio = np.zeros(16000, dtype="float32")

        # First call raises; second returns a real transcript
        transcribe = AsyncMock(side_effect=[SpeechToTextError("mic failed"), "hello rex"])
        assistant = MagicMock()
        assistant.generate_reply = AsyncMock(return_value="Hi")
        speak = AsyncMock()

        loop = VoiceLoop(
            assistant,
            wake_listener=wake_listener,
            detection_source=AsyncMock(return_value=audio),
            record_phrase=AsyncMock(return_value=audio),
            transcribe=transcribe,
            speak=speak,
        )
        asyncio.run(loop.run())
        # Second cycle succeeded, speak was called once
        speak.assert_awaited_once()


# ---------------------------------------------------------------------------
# 3. LLM response generated
# ---------------------------------------------------------------------------


class TestLLMResponseGenerated:
    def test_generate_reply_called_with_transcript(self):
        """LLM generate_reply() receives the transcript text."""
        loop, assistant, _, _, _ = _make_voice_loop(transcript="what time is it")
        asyncio.run(loop.run(max_interactions=1))
        assistant.generate_reply.assert_awaited_once_with("what time is it", voice_mode=True)

    def test_response_passed_to_speak(self):
        """LLM response is forwarded to speak()."""
        speak = AsyncMock()
        loop, _, _, _, _ = _make_voice_loop(
            transcript="play music",
            llm_response="Playing music now",
            speak_spy=speak,
        )
        asyncio.run(loop.run(max_interactions=1))
        # The loop appends a period if missing
        speak.assert_awaited_once_with("Playing music now.")

    def test_response_already_ending_with_period_not_doubled(self):
        """LLM response ending with period is not double-punctuated."""
        speak = AsyncMock()
        loop, _, _, _, _ = _make_voice_loop(
            transcript="hello",
            llm_response="Hello there.",
            speak_spy=speak,
        )
        asyncio.run(loop.run(max_interactions=1))
        speak.assert_awaited_once_with("Hello there.")

    def test_llm_error_does_not_crash_loop(self):
        """Unexpected error from LLM is caught and loop survives."""
        audio = np.zeros(16000, dtype="float32")
        assistant = MagicMock()
        assistant.generate_reply = AsyncMock(side_effect=RuntimeError("LLM down"))
        speak = AsyncMock()

        loop = VoiceLoop(
            assistant,
            wake_listener=_make_wake_listener(trigger_count=1),
            detection_source=AsyncMock(return_value=audio),
            record_phrase=AsyncMock(return_value=audio),
            transcribe=AsyncMock(return_value="hello"),
            speak=speak,
        )
        # Should not raise
        asyncio.run(loop.run(max_interactions=1))
        speak.assert_not_awaited()


# ---------------------------------------------------------------------------
# 4. Response spoken aloud
# ---------------------------------------------------------------------------


class TestResponseSpokenAloud:
    def test_speak_called_once_per_interaction(self):
        """speak() is called exactly once per successful interaction."""
        speak = AsyncMock()
        loop, _, _, _, _ = _make_voice_loop(speak_spy=speak)
        asyncio.run(loop.run(max_interactions=1))
        speak.assert_awaited_once()

    def test_speak_receives_llm_output(self):
        """speak() is called with the LLM-generated text."""
        speak = AsyncMock()
        loop, _, _, _, _ = _make_voice_loop(
            llm_response="The weather is sunny",
            speak_spy=speak,
        )
        asyncio.run(loop.run(max_interactions=1))
        call_arg = speak.await_args[0][0]
        assert "sunny" in call_arg

    def test_full_pipeline_end_to_end(self):
        """Complete pipeline: wake → STT → LLM → TTS all execute in sequence."""
        spoken_text: list[str] = []

        async def capture_speak(text: str) -> None:
            spoken_text.append(text)

        audio = np.zeros(16000, dtype="float32")
        transcript = "turn on the lights"
        llm_response = "Turning on the lights"

        assistant = MagicMock()
        assistant.generate_reply = AsyncMock(return_value=llm_response)
        transcribe = AsyncMock(return_value=transcript)
        record_phrase = AsyncMock(return_value=audio)

        loop = VoiceLoop(
            assistant,
            wake_listener=_make_wake_listener(trigger_count=1),
            detection_source=AsyncMock(return_value=audio),
            record_phrase=record_phrase,
            transcribe=transcribe,
            speak=capture_speak,
        )
        asyncio.run(loop.run(max_interactions=1))

        assert len(spoken_text) == 1
        assert "Turning on the lights" in spoken_text[0]
        transcribe.assert_awaited_once_with(audio)
        assistant.generate_reply.assert_awaited_once_with(transcript, voice_mode=True)
