#!/usr/bin/env python3
"""Diagnostic script: trigger the full voice pipeline with mocks and capture PIPELINE trace.

Usage:
    python scripts/test_voice_pipeline.py

Output:
    Prints each [PIPELINE] stage to stdout and writes pipeline_trace.log.
    The analysis section identifies the last completed stage and first missing stage,
    which is the confirmed break point for voice audio delivery.

Run with LOG_LEVEL=DEBUG (the script sets it automatically).
"""
from __future__ import annotations

import asyncio
import sys
import os as _os

# Ensure project root is on sys.path so voice_loop is importable when running
# from any working directory (e.g. python scripts/test_voice_pipeline.py).
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import io
import logging
import os
import struct
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Force DEBUG logging before any rex imports so pipeline logs are captured.
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ.setdefault("REX_TESTING", "1")

# ── log capture ────────────────────────────────────────────────────────────────
log_stream = io.StringIO()
_fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s  %(message)s")

_cap = logging.StreamHandler(log_stream)
_cap.setLevel(logging.DEBUG)
_cap.setFormatter(_fmt)

_out = logging.StreamHandler(sys.stdout)
_out.setLevel(logging.DEBUG)
_out.setFormatter(_fmt)

root = logging.getLogger()
root.setLevel(logging.DEBUG)
root.addHandler(_cap)
root.addHandler(_out)

# ── pipeline stages we expect ──────────────────────────────────────────────────
EXPECTED_STAGES = [
    "llm_response_received",
    "tts_input_prepared",
    "tts_engine_called",
    "audio_data_returned",
    "audio_playback_initiated",
    "audio_playback_completed",
]


def _make_wav(path: str) -> None:
    """Write a minimal valid WAV file so soundfile.read() succeeds."""
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(struct.pack(f"<{22050}h", *([0] * 22050)))


async def run_mock_pipeline() -> str:
    """Run _process_conversation with all heavy deps mocked.

    Returns the captured log output as a string.
    """
    import numpy as np

    # Import after env vars are set.
    from rex.config import AppConfig
    from voice_loop import AsyncRexAssistant

    config = AppConfig()

    # ── TTS mock ──────────────────────────────────────────────────────────────
    mock_tts = MagicMock()

    def fake_tts_to_file(*args, **kwargs):
        path = kwargs.get("file_path")
        if path:
            _make_wav(path)

    mock_tts.tts_to_file.side_effect = fake_tts_to_file

    # ── whisper mock ──────────────────────────────────────────────────────────
    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = {"text": "hello rex"}

    # ── LLM mock ──────────────────────────────────────────────────────────────
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Hello! I am Rex, your AI assistant. How can I help?"

    # ── audio playback mock ───────────────────────────────────────────────────
    # Patch winsound so we don't need a real audio device on Windows.
    mock_winsound = MagicMock()

    fake_audio = np.zeros(16000, dtype=np.float32)

    # Build assistant without running __init__ so we can inject mocks directly.
    assistant = AsyncRexAssistant.__new__(AsyncRexAssistant)
    assistant.config = config
    assistant.language_model = mock_llm
    assistant._tts = mock_tts
    assistant._whisper_model = mock_whisper
    assistant._wake_model = MagicMock()
    assistant._wake_keyword = "hey_rex"
    assistant._sample_rate = 16000
    assistant.active_user = "james"
    assistant.user_voice_refs = {"james": None}
    assistant.plugins = {}
    assistant._wake_sound_path = None
    assistant._running = True
    assistant._state = "running"

    with (
        patch("voice_loop.append_history_entry"),
        patch("voice_loop.export_transcript"),
        patch.object(assistant, "_record_audio", return_value=fake_audio),
        patch.object(assistant, "transcribe", new=AsyncMock(return_value="hello rex")),
        patch.object(assistant, "_play_wake_sound"),
        patch.dict("sys.modules", {"winsound": mock_winsound}),
        # Patch _get_tts so it returns our mock without loading XTTS.
        patch.object(type(assistant), "_get_tts", lambda self: mock_tts),
    ):
        print("\n" + "=" * 60)
        print("RUNNING VOICE PIPELINE — MOCK MODE")
        print("=" * 60 + "\n")

        try:
            await assistant._process_conversation()
            print("\n" + "=" * 60)
            print("PIPELINE RUN COMPLETE")
            print("=" * 60)
        except Exception as exc:
            print("\n" + "=" * 60)
            print(f"PIPELINE RAISED: {type(exc).__name__}: {exc}")
            print("=" * 60)

    return log_stream.getvalue()


def analyze_trace(log_output: str) -> dict:
    """Return which expected stages were and were not logged."""
    completed = [s for s in EXPECTED_STAGES if f"stage={s}" in log_output]
    missing = [s for s in EXPECTED_STAGES if f"stage={s}" not in log_output]
    return {
        "completed": completed,
        "missing": missing,
        "last_completed": completed[-1] if completed else None,
        "first_missing": missing[0] if missing else None,
        "all_passed": len(missing) == 0,
    }


if __name__ == "__main__":
    log_output = asyncio.run(run_mock_pipeline())

    trace_path = Path("pipeline_trace.log")
    trace_path.write_text(log_output, encoding="utf-8")

    result = analyze_trace(log_output)

    print("\n" + "=" * 60)
    print("PIPELINE TRACE ANALYSIS")
    print("=" * 60)
    print(f"Stages completed : {result['completed']}")
    print(f"Stages missing   : {result['missing']}")
    print(f"Last completed   : {result['last_completed']}")
    print(f"First missing    : {result['first_missing']}")
    print(f"Full trace       : {trace_path.resolve()}")

    if result["all_passed"]:
        print("\n✓ ALL 6 STAGES COMPLETED — pipeline is functioning with mocks.")
    else:
        print(
            f"\n✗ BREAK POINT detected: after '{result['last_completed']}' "
            f"before '{result['first_missing']}'"
        )
    sys.exit(0 if result["all_passed"] else 1)
