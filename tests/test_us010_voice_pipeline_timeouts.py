"""US-010: Voice pipeline timeout and recovery tests."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("numpy")
import numpy as np  # noqa: E402

from rex.voice_loop import VoiceLoop  # noqa: E402


def _make_loop(
    *,
    stt_timeout: float = 30.0,
    llm_timeout: float = 60.0,
    tts_timeout: float = 30.0,
    transcribe=None,
    speak=None,
    generate_reply=None,
):
    """Build a VoiceLoop with controllable stubs."""
    if transcribe is None:
        transcribe = AsyncMock(return_value="hello")
    if speak is None:
        speak = AsyncMock(return_value=None)

    assistant = MagicMock()
    if generate_reply is not None:
        assistant.generate_reply = generate_reply
    else:
        assistant.generate_reply = AsyncMock(return_value="hi there")
    # Remove stream_reply so the non-streaming path is used
    del assistant.stream_reply

    audio_frame = np.zeros(16000, dtype=np.float32)

    # Wake listener that yields exactly one detection then stops
    class _SingleWake:
        async def listen(self, _source):
            yield None

    return (
        VoiceLoop(
            assistant,
            wake_listener=_SingleWake(),
            detection_source=AsyncMock(return_value=audio_frame),
            record_phrase=AsyncMock(return_value=audio_frame),
            transcribe=transcribe,
            speak=speak,
            stt_timeout=stt_timeout,
            llm_timeout=llm_timeout,
            tts_timeout=tts_timeout,
        ),
        assistant,
    )


async def _slow_coro(*_args, **_kwargs):
    """A coroutine that never finishes (simulates a hang)."""
    await asyncio.sleep(9999)


# ---------------------------------------------------------------------------
# STT timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio_mode_not_required
def test_stt_timeout_logs_and_recovers(caplog):
    """A hung STT call times out, logs pipeline_timeout, and the loop exits cleanly."""
    loop, _assistant = _make_loop(stt_timeout=0.05, transcribe=AsyncMock(side_effect=_slow_coro))

    with patch("rex.voice_loop.settings"):
        asyncio.run(loop.run(max_interactions=1))

    events = [r.message for r in caplog.records]
    assert any(
        "STT stage timed out" in m for m in events
    ), f"Expected timeout message, got: {events}"


def test_stt_timeout_event_field(caplog):
    """The timeout log record has event=pipeline_timeout and stage=stt."""
    loop, _assistant = _make_loop(stt_timeout=0.05, transcribe=AsyncMock(side_effect=_slow_coro))

    with patch("rex.voice_loop.settings"):
        asyncio.run(loop.run(max_interactions=1))

    timeout_records = [r for r in caplog.records if getattr(r, "event", None) == "pipeline_timeout"]
    assert timeout_records, "Expected a pipeline_timeout log record"
    assert timeout_records[0].stage == "stt"


# ---------------------------------------------------------------------------
# LLM timeout
# ---------------------------------------------------------------------------


def test_llm_timeout_logs_and_recovers(caplog):
    """A hung LLM generate_reply call times out, logs pipeline_timeout, loop exits cleanly."""
    loop, _assistant = _make_loop(
        llm_timeout=0.05,
        generate_reply=AsyncMock(side_effect=_slow_coro),
    )

    with patch("rex.voice_loop.settings"):
        asyncio.run(loop.run(max_interactions=1))

    events = [r.message for r in caplog.records]
    assert any(
        "LLM stage timed out" in m for m in events
    ), f"Expected timeout message, got: {events}"


def test_llm_timeout_stage_field(caplog):
    """LLM timeout log record has stage=llm."""
    loop, _assistant = _make_loop(
        llm_timeout=0.05,
        generate_reply=AsyncMock(side_effect=_slow_coro),
    )

    with patch("rex.voice_loop.settings"):
        asyncio.run(loop.run(max_interactions=1))

    timeout_records = [r for r in caplog.records if getattr(r, "event", None) == "pipeline_timeout"]
    assert timeout_records
    assert timeout_records[0].stage == "llm"


# ---------------------------------------------------------------------------
# TTS timeout
# ---------------------------------------------------------------------------


def test_tts_timeout_logs_and_recovers(caplog):
    """A hung TTS speak call times out, logs pipeline_timeout, loop exits cleanly."""
    loop, _assistant = _make_loop(
        tts_timeout=0.05,
        speak=AsyncMock(side_effect=_slow_coro),
    )

    with patch("rex.voice_loop.settings"):
        asyncio.run(loop.run(max_interactions=1))

    events = [r.message for r in caplog.records]
    assert any(
        "TTS stage timed out" in m for m in events
    ), f"Expected timeout message, got: {events}"


def test_tts_timeout_stage_field(caplog):
    """TTS timeout log record has stage=tts."""
    loop, _assistant = _make_loop(
        tts_timeout=0.05,
        speak=AsyncMock(side_effect=_slow_coro),
    )

    with patch("rex.voice_loop.settings"):
        asyncio.run(loop.run(max_interactions=1))

    timeout_records = [r for r in caplog.records if getattr(r, "event", None) == "pipeline_timeout"]
    assert timeout_records
    assert timeout_records[0].stage == "tts"


# ---------------------------------------------------------------------------
# Happy path — all stages complete within timeout
# ---------------------------------------------------------------------------


def test_happy_path_no_timeout(caplog):
    """When all stages complete quickly, no timeout event is logged."""
    loop, _assistant = _make_loop()

    with patch("rex.voice_loop.settings"):
        asyncio.run(loop.run(max_interactions=1))

    timeout_records = [r for r in caplog.records if getattr(r, "event", None) == "pipeline_timeout"]
    assert not timeout_records, f"Unexpected timeout records: {timeout_records}"
