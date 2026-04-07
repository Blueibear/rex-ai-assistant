"""Tests for US-074: Diagnose and fix standalone rex_loop.py voice conversation.

Acceptance Criteria:
- run rex_loop.py and document which stages succeed (pipeline_stage_ok events)
- for each failing stage, add a structured log message (pipeline_stage_failed events)
- fix all identified failures so a full cycle completes
- rex_loop.py uses build_voice_loop from rex.voice_loop (canonical implementation)
- if a stage cannot be fixed, the loop logs the blocker and exits cleanly
- integration test with mocked audio confirms full pipeline completion
- Typecheck passes
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from rex.assistant_errors import AudioDeviceError
from rex.voice_loop import VoiceLoop

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False

_requires_numpy = pytest.mark.skipif(not _NUMPY_AVAILABLE, reason="numpy not installed")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _OnceListener:
    """Yield a single detection frame then stop — used to run the loop once."""

    def __init__(self) -> None:
        self._fired = False

    async def listen(self, source):
        if not self._fired:
            self._fired = True
            yield await source()

    def stop(self) -> None:
        pass


def _make_pipeline(
    transcribe_result: str = "hello rex",
    llm_response: str = "Hi there!",
    sample_rate: int = 16000,
):
    """Build a minimal VoiceLoop with mocked callables for all pipeline stages.

    Requires numpy — callers must guard with ``@_requires_numpy``.
    """
    audio = np.ones(sample_rate, dtype=np.float32)  # type: ignore[union-attr]

    async def _detection_source():
        return np.ones(4, dtype=np.float32)  # type: ignore[union-attr]

    async def _record_phrase():
        return audio

    async def _transcribe(_audio):
        return transcribe_result

    spoken_texts: list[str | None] = []

    async def _speak(text):
        spoken_texts.append(text)

    assistant = MagicMock()
    assistant.generate_reply = AsyncMock(return_value=llm_response)
    del assistant.stream_reply  # force non-streaming path

    loop = VoiceLoop(
        assistant,
        wake_listener=_OnceListener(),
        detection_source=_detection_source,
        record_phrase=_record_phrase,
        transcribe=_transcribe,
        speak=_speak,
        sample_rate=sample_rate,
    )
    return loop, assistant, spoken_texts


# ---------------------------------------------------------------------------
# AC: integration test with mocked audio confirms full pipeline completion
# ---------------------------------------------------------------------------


@_requires_numpy
def test_full_pipeline_happy_path(caplog):
    """Full pipeline: wake → capture → STT → LLM → TTS with mocked audio."""
    loop, assistant, spoken_texts = _make_pipeline(
        transcribe_result="what is the weather",
        llm_response="It looks sunny today.",
    )

    with caplog.at_level(logging.DEBUG, logger="rex.voice_loop"):
        asyncio.run(loop.run(max_interactions=1))

    events = [getattr(r, "event", None) for r in caplog.records]
    assert "stt_handoff" in events, "Expected stt_handoff log event"
    assert "audio_capture_complete" in events, "Expected audio_capture_complete log event"

    assistant.generate_reply.assert_awaited_once()
    call_args = assistant.generate_reply.call_args
    assert call_args[0][0] == "what is the weather"

    assert spoken_texts, "speak() must be called with LLM response"
    assert "It looks sunny today." in spoken_texts[0]


@_requires_numpy
def test_full_pipeline_completes_without_hanging():
    """Pipeline terminates cleanly after one interaction (no hang)."""
    loop, _assistant, _spoken = _make_pipeline()
    asyncio.run(loop.run(max_interactions=1))


@_requires_numpy
def test_stt_stage_log_contains_audio_samples(caplog):
    """STT handoff log event is emitted with audio_samples field."""
    loop, _assistant, _spoken = _make_pipeline(sample_rate=16000)

    with caplog.at_level(logging.DEBUG, logger="rex.voice_loop"):
        asyncio.run(loop.run(max_interactions=1))

    stt_events = [r for r in caplog.records if getattr(r, "event", None) == "stt_handoff"]
    assert stt_events, "Expected stt_handoff event"
    assert hasattr(stt_events[0], "audio_samples"), "stt_handoff must include audio_samples field"


# ---------------------------------------------------------------------------
# AC: if a stage cannot be fixed (missing hardware), loop logs blocker and
# exits cleanly
# ---------------------------------------------------------------------------


def test_audio_device_error_logs_pipeline_blocker(caplog):
    """AudioDeviceError from the detection source is logged as pipeline_blocker."""

    class _BrokenListener:
        async def listen(self, source):
            raise AudioDeviceError("no microphone detected")
            yield  # pragma: no cover — makes this an async generator

        def stop(self) -> None:
            pass

    async def _detection_source():  # pragma: no cover
        return None

    async def _record_phrase():  # pragma: no cover
        return None

    async def _transcribe(_audio):  # pragma: no cover
        return "text"

    async def _speak(_text):  # pragma: no cover
        pass

    assistant = MagicMock()
    assistant.generate_reply = AsyncMock(return_value="ok")
    del assistant.stream_reply

    loop = VoiceLoop(
        assistant,
        wake_listener=_BrokenListener(),
        detection_source=_detection_source,
        record_phrase=_record_phrase,
        transcribe=_transcribe,
        speak=_speak,
    )

    with caplog.at_level(logging.ERROR, logger="rex.voice_loop"):
        asyncio.run(loop.run())

    blocker_records = [r for r in caplog.records if getattr(r, "event", None) == "pipeline_blocker"]
    assert blocker_records, "Expected pipeline_blocker log event on AudioDeviceError"
    assert getattr(blocker_records[0], "stage", None) == "audio_device"


# ---------------------------------------------------------------------------
# AC: rex_loop.py uses build_voice_loop from rex.voice_loop (canonical)
# ---------------------------------------------------------------------------


def test_rex_loop_imports_build_voice_loop_from_canonical_module():
    """rex_loop.py must import build_voice_loop from rex.voice_loop (not root voice_loop)."""
    import rex_loop

    source = inspect.getsource(rex_loop)
    assert "from rex.voice_loop import build_voice_loop" in source, (
        "rex_loop.py must use 'from rex.voice_loop import build_voice_loop' "
        "(the canonical implementation)"
    )


# ---------------------------------------------------------------------------
# AC: pipeline stage log events emitted during a full run cycle
# ---------------------------------------------------------------------------


@_requires_numpy
def test_pipeline_stage_events_during_run(caplog):
    """All key pipeline stage log events appear during a full run cycle."""
    loop, _assistant, _spoken = _make_pipeline()

    with caplog.at_level(logging.DEBUG, logger="rex.voice_loop"):
        asyncio.run(loop.run(max_interactions=1))

    events = {getattr(r, "event", None) for r in caplog.records}
    assert "audio_capture_complete" in events, "Missing audio_capture_complete event"
    assert "stt_handoff" in events, "Missing stt_handoff event"
