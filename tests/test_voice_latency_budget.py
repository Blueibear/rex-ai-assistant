from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from rex.voice_loop import VoiceLoop

VOICE_PIPELINE_DOC = Path(__file__).parent.parent / "docs" / "voice_pipeline.md"

STAGE_BUDGETS_MS = {
    "activation_to_capture": 250.0,
    "capture": 500.0,
    "stt": 500.0,
    "llm": 500.0,
    "tts_playback": 500.0,
    "total": 2000.0,
}


class _OnceListener:
    async def listen(self, _source):
        yield b"wake"

    def mark_listening_started(self, *, reason: str = "test") -> None:
        return None

    def reset(self, *, reason: str = "test") -> None:
        return None


def _run_synthetic_pipeline(caplog) -> dict[str, logging.LogRecord]:
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

    expected = {
        "wake_detected",
        "capture_started",
        "capture_ended",
        "stt_completed",
        "llm_completed",
        "playback_completed",
    }
    records = {
        record.event: record
        for record in caplog.records
        if getattr(record, "event", None) in expected
    }
    assert records.keys() == expected
    return records


def test_synthetic_voice_pipeline_stays_within_stage_budgets(caplog) -> None:
    records = _run_synthetic_pipeline(caplog)
    wake = records["wake_detected"]
    capture_started = records["capture_started"]
    playback = records["playback_completed"]

    measured_ms = {
        "activation_to_capture": (capture_started.start_ns - wake.start_ns) / 1_000_000,
        "capture": records["capture_ended"].duration_ms,
        "stt": records["stt_completed"].duration_ms,
        "llm": records["llm_completed"].duration_ms,
        "tts_playback": playback.duration_ms,
        "total": ((playback.start_ns - wake.start_ns) / 1_000_000) + playback.duration_ms,
    }

    for stage, budget_ms in STAGE_BUDGETS_MS.items():
        assert measured_ms[stage] <= budget_ms, (
            f"{stage} latency {measured_ms[stage]:.3f} ms exceeded "
            f"the synthetic CI budget of {budget_ms:.0f} ms"
        )


def test_documented_budget_table_matches_enforced_contract() -> None:
    document = VOICE_PIPELINE_DOC.read_text(encoding="utf-8")
    labels = {
        "activation_to_capture": "Activation accepted -> capture start",
        "capture": "Capture callback",
        "stt": "STT callback",
        "llm": "LLM response callback",
        "tts_playback": "TTS/playback callback",
        "total": "Activation -> playback complete",
    }

    for stage, label in labels.items():
        budget_ms = STAGE_BUDGETS_MS[stage]
        expected_row = f"| {label} | `{stage}` | {budget_ms:.0f} ms |"
        assert expected_row in document
