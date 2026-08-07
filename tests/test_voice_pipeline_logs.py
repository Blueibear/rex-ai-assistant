from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock

from rex.logging_utils import JsonFormatter
from rex.voice_loop import VoiceLoop

EXPECTED_EVENTS = [
    "wake_detected",
    "capture_started",
    "capture_ended",
    "stt_started",
    "stt_completed",
    "llm_started",
    "llm_completed",
    "tts_started",
    "playback_completed",
]


class _OnceListener:
    async def listen(self, _source):
        yield b"wake"

    def mark_listening_started(self, *, reason: str = "test") -> None:
        return None

    def reset(self, *, reason: str = "test") -> None:
        return None


def test_happy_path_emits_structured_voice_pipeline_events(caplog) -> None:
    async def detection_source():
        return b"wake"

    async def record_phrase():
        return b"audio"

    async def transcribe(_audio):
        return "hello rex"

    async def speak(_text: str) -> None:
        return None

    assistant = MagicMock()
    assistant.generate_reply = AsyncMock(return_value="Hello there.")
    del assistant.stream_reply

    loop = VoiceLoop(
        assistant,
        wake_listener=_OnceListener(),
        detection_source=detection_source,
        record_phrase=record_phrase,
        transcribe=transcribe,
        speak=speak,
        post_interaction_cooldown=0,
    )

    with caplog.at_level(logging.INFO, logger="rex.voice_loop"):
        asyncio.run(loop.run(max_interactions=1))

    records = [r for r in caplog.records if getattr(r, "event", None) in EXPECTED_EVENTS]
    assert [r.event for r in records] == EXPECTED_EVENTS

    session_ids = {r.session_id for r in records}
    assert len(session_ids) == 1
    assert next(iter(session_ids))

    for record in records:
        assert isinstance(record.start_ns, int)
        assert record.start_ns > 0

        payload = json.loads(JsonFormatter().format(record))
        assert payload["extra"]["event"] == record.event
        assert payload["extra"]["session_id"] == record.session_id
        assert payload["extra"]["start_ns"] == record.start_ns

    completed = {
        "capture_ended",
        "stt_completed",
        "llm_completed",
        "playback_completed",
    }
    for record in records:
        if record.event in completed:
            assert isinstance(record.duration_ms, float)
            assert record.duration_ms >= 0.0
        else:
            assert not hasattr(record, "duration_ms")


def test_streaming_happy_path_emits_same_canonical_event_set(caplog) -> None:
    async def detection_source():
        return b"wake"

    async def record_phrase():
        return b"audio"

    async def transcribe(_audio):
        return "hello rex"

    async def speak(_text: str) -> None:
        raise AssertionError("non-streaming speak should not be used")

    async def stream_reply(_text: str, *, voice_mode: bool = False):
        assert voice_mode is True
        yield "Hello "
        yield "there."

    async def speak_streaming(chunks) -> None:
        async for _chunk in chunks:
            pass

    assistant = MagicMock()
    assistant.generate_reply = AsyncMock()
    assistant.stream_reply = stream_reply

    loop = VoiceLoop(
        assistant,
        wake_listener=_OnceListener(),
        detection_source=detection_source,
        record_phrase=record_phrase,
        transcribe=transcribe,
        speak=speak,
        speak_streaming=speak_streaming,
        post_interaction_cooldown=0,
    )

    with caplog.at_level(logging.INFO, logger="rex.voice_loop"):
        asyncio.run(loop.run(max_interactions=1))

    records = [r for r in caplog.records if getattr(r, "event", None) in EXPECTED_EVENTS]
    assert {r.event for r in records} == set(EXPECTED_EVENTS)
    assert len(records) == len(EXPECTED_EVENTS)
    assistant.generate_reply.assert_not_awaited()

    streaming_completed = {
        r.event: r for r in records if r.event in {"llm_completed", "playback_completed"}
    }
    assert streaming_completed["llm_completed"].timing_scope == "streaming_llm_tts_playback"
    assert streaming_completed["playback_completed"].timing_scope == "streaming_llm_tts_playback"
