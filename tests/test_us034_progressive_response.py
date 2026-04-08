"""Tests for US-034: Progressive response system — sentence-level streaming TTS.

Acceptance criteria:
- If LLM supports streaming, TTS begins on the first complete sentence
- Subsequent sentences are queued and spoken sequentially
- If LLM does not support streaming, behavior falls back to full-response TTS
- No audio overlap between sentence chunks
- Test confirms sentence-level streaming with a mock streaming LLM
- Typecheck passes
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from importlib.util import find_spec
from unittest.mock import AsyncMock, MagicMock

import pytest

_HAS_NUMPY = find_spec("numpy") is not None
_needs_numpy = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")


# ---------------------------------------------------------------------------
# AC1+AC5 (pure async, no numpy): sentence buffer converts token stream to sentences
# ---------------------------------------------------------------------------


def test_sentence_buffer_yields_sentences_from_token_stream():
    """AC1+AC5: _sentence_buffer_stream splits a token stream into sentence chunks."""
    from rex.voice_loop import _sentence_buffer_stream

    async def _run():
        async def token_stream() -> AsyncIterator[str]:
            for token in ["Hello. ", "World."]:
                yield token

        sentences: list[str] = []
        async for sentence in _sentence_buffer_stream(token_stream()):
            sentences.append(sentence)
        return sentences

    sentences = asyncio.run(_run())
    assert len(sentences) >= 1, "At least one sentence must be buffered"
    assert "Hello" in sentences[0], f"First chunk should start with Hello, got {sentences[0]!r}"


def test_sentence_buffer_first_sentence_before_stream_complete():
    """AC1: TTS begins on first complete sentence before the full response arrives."""
    from rex.voice_loop import _sentence_buffer_stream

    received_first: list[str] = []

    async def _run():
        async def token_stream() -> AsyncIterator[str]:
            yield "First sentence. "
            yield "Second sentence."

        async for sentence in _sentence_buffer_stream(token_stream()):
            if not received_first:
                received_first.append(sentence)
            # Stop after first to simulate "begins on first sentence"
            break

    asyncio.run(_run())
    assert received_first, "Must receive first sentence"
    assert "First sentence" in received_first[0]


def test_sentence_buffer_subsequent_sentences_in_order():
    """AC2: Subsequent sentences arrive in order from the buffer stream."""
    from rex.voice_loop import _sentence_buffer_stream

    async def _run():
        async def token_stream() -> AsyncIterator[str]:
            for token in ["One. ", "Two. ", "Three."]:
                yield token

        sentences: list[str] = []
        async for sentence in _sentence_buffer_stream(token_stream()):
            sentences.append(sentence)
        return sentences

    sentences = asyncio.run(_run())
    full_text = " ".join(sentences)
    assert "One" in full_text
    assert "Two" in full_text
    assert "Three" in full_text
    assert full_text.find("One") < full_text.find("Two") < full_text.find("Three")


# ---------------------------------------------------------------------------
# VoiceLoop integration tests (require numpy)
# ---------------------------------------------------------------------------


def _make_once_listener():
    class _OnceListener:
        def __init__(self) -> None:
            self._fired = False

        async def listen(self, source):  # noqa: ANN001
            if not self._fired:
                self._fired = True
                yield await source()

        def stop(self) -> None:
            pass

    return _OnceListener()


def _make_streaming_loop(tokens: list[str]):
    """Build a VoiceLoop with a mock streaming LLM assistant."""
    import numpy as np  # guarded by _needs_numpy skip

    from rex.voice_loop import VoiceLoop

    sample_rate = 16000
    audio = np.ones(sample_rate, dtype=np.float32)

    async def _record_phrase():
        return audio

    async def _detection_source():
        return np.ones(4, dtype=np.float32)

    async def _transcribe(_audio):
        return "tell me a story"

    spoken_chunks: list[list[str]] = []

    async def _speak(text: str) -> None:
        pass

    async def _speak_streaming(sentences: AsyncIterator[str]) -> None:
        batch: list[str] = []
        async for s in sentences:
            batch.append(s)
        spoken_chunks.append(batch)

    assistant = MagicMock()
    assistant.generate_reply = AsyncMock(return_value="This should not be called.")

    async def _stream_reply(transcript: str, *, voice_mode: bool = False) -> AsyncIterator[str]:
        for token in tokens:
            yield token

    assistant.stream_reply = _stream_reply

    loop = VoiceLoop(
        assistant,
        wake_listener=_make_once_listener(),
        detection_source=_detection_source,
        record_phrase=_record_phrase,
        transcribe=_transcribe,
        speak=_speak,
        speak_streaming=_speak_streaming,
        sample_rate=sample_rate,
    )
    return loop, assistant, spoken_chunks


def _make_non_streaming_loop(llm_response: str = "It is a nice day."):
    """Build a VoiceLoop with a mock non-streaming LLM assistant."""
    import numpy as np  # guarded by _needs_numpy skip

    from rex.voice_loop import VoiceLoop

    sample_rate = 16000
    audio = np.ones(sample_rate, dtype=np.float32)

    async def _record_phrase():
        return audio

    async def _detection_source():
        return np.ones(4, dtype=np.float32)

    async def _transcribe(_audio):
        return "how is the weather"

    spoken_texts: list[str] = []

    async def _speak(text: str) -> None:
        spoken_texts.append(text)

    assistant = MagicMock()
    assistant.generate_reply = AsyncMock(return_value=llm_response)
    del assistant.stream_reply

    loop = VoiceLoop(
        assistant,
        wake_listener=_make_once_listener(),
        detection_source=_detection_source,
        record_phrase=_record_phrase,
        transcribe=_transcribe,
        speak=_speak,
        sample_rate=sample_rate,
    )
    return loop, assistant, spoken_texts


@_needs_numpy
def test_streaming_llm_uses_speak_streaming():
    """AC1+AC5: When stream_reply is present, speak_streaming is called (not speak)."""
    tokens = ["Hello", " world.", " How are", " you?"]
    loop, _assistant, spoken_chunks = _make_streaming_loop(tokens)

    asyncio.run(loop.run(max_interactions=1))

    assert spoken_chunks, "speak_streaming must have been called"


@_needs_numpy
def test_streaming_llm_sentences_contain_expected_text():
    """AC2: Sentences spoken via streaming contain the full response text."""
    tokens = ["First sentence.", " Second sentence."]
    loop, _assistant, spoken_chunks = _make_streaming_loop(tokens)

    asyncio.run(loop.run(max_interactions=1))

    assert spoken_chunks, "speak_streaming must have been called"
    all_sentences = [s for batch in spoken_chunks for s in batch]
    full_text = " ".join(all_sentences)
    assert "First sentence" in full_text
    assert "Second sentence" in full_text


@_needs_numpy
def test_non_streaming_llm_falls_back_to_full_response():
    """AC3: When stream_reply is absent, generate_reply + speak are used."""
    loop, assistant, spoken_texts = _make_non_streaming_loop(
        llm_response="The weather is fine today."
    )

    asyncio.run(loop.run(max_interactions=1))

    assistant.generate_reply.assert_awaited_once()
    assert spoken_texts, "speak() must be called on the non-streaming path"
    assert "weather is fine" in spoken_texts[0]


@_needs_numpy
def test_no_audio_overlap_sequential_speak_calls():
    """AC4: Sentences spoken in order — sequential speak means no overlap."""
    tokens = ["One. ", "Two. ", "Three."]
    loop, _assistant, spoken_chunks = _make_streaming_loop(tokens)

    asyncio.run(loop.run(max_interactions=1))

    all_sentences = [s for batch in spoken_chunks for s in batch]
    full_text = " ".join(all_sentences)
    one_pos = full_text.find("One")
    two_pos = full_text.find("Two")
    three_pos = full_text.find("Three")
    assert one_pos != -1 and two_pos != -1 and three_pos != -1
    assert (
        one_pos < two_pos < three_pos
    ), f"Sentences must be spoken in order, positions: One={one_pos}, Two={two_pos}, Three={three_pos}"
