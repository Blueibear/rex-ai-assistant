#!/usr/bin/env python3
"""Deterministic privacy-safe Rex latency baseline harness."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rex.assistant as assistant_module  # noqa: E402
from rex.model_router import ModelRouter  # noqa: E402
from rex.rexbench import BenchmarkSample, build_report  # noqa: E402
from rex.voice_loop import VoiceLoop  # noqa: E402

REQUEST_CLASSES = (
    "typed_chat",
    "voice",
    "read_only_tool",
    "mutating_tool",
    "unavailable_capability",
)


@dataclass
class _Routing:
    default: str = "bench-model"
    coding: str = ""
    reasoning: str = ""
    search: str = ""
    vision: str = ""
    fast: str = ""


@dataclass
class _Settings:
    llm_provider: str = "transformers"
    llm_model: str = "bench-model"
    llm_max_tokens: int = 10
    llm_temperature: float = 0.7
    llm_top_p: float = 0.9
    llm_top_k: int = 50
    llm_seed: int = 42
    max_memory_items: int = 5
    transcripts_dir: str = "transcripts"
    persist_history: bool = False
    followups_enabled: bool = False
    ha_base_url: str | None = None
    ha_token: str | None = None
    user_id: str = "benchmark"
    active_profile: str = "default"
    response_cache_ttl: int = 0
    model_routing: _Routing = field(default_factory=_Routing)


class _BenchLLM:
    def __init__(self, response: str = "benchmark reply") -> None:
        self.model_name = "bench-model"
        self.response = response

    def generate(self, prompt=None, *, messages=None, config=None):
        time.sleep(0.002)
        return self.response

    def stream(self, prompt=None, *, messages=None, config=None):
        time.sleep(0.002)
        yield self.response


class _BenchToolDispatcher:
    def __init__(self, operation: str) -> None:
        self.operation = operation

    def select_tools(self, transcript):
        return [SimpleNamespace(operation=self.operation)]

    def execute_tools(self, selected, transcript, *, user_id):
        time.sleep(0.002)
        return []

    def format_tool_context(self, results):
        return "benchmark result"


class _RecordCapture(logging.Handler):
    def __init__(self, event: str) -> None:
        super().__init__()
        self.event = event
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "event", None) == self.event:
            self.records.append(record)


def _make_assistant(request_class: str):
    response = (
        "Capability unavailable."
        if request_class == "unavailable_capability"
        else "benchmark reply"
    )
    llm = _BenchLLM(response=response)

    class _Factory:
        def __new__(cls, *args, **kwargs):
            return llm

    started = time.perf_counter_ns()
    with (
        patch.object(assistant_module, "LanguageModel", _Factory),
        patch.object(ModelRouter, "_fetch_ollama_models", lambda self: None),
    ):
        assistant = assistant_module.Assistant(
            settings_obj=_Settings(), transcripts_dir="transcripts", user_id="benchmark"
        )
    startup_ms = (time.perf_counter_ns() - started) / 1_000_000
    assistant._router._available_ollama_models = {"bench-model"}

    tool_dispatcher = None
    if request_class == "read_only_tool":
        tool_dispatcher = _BenchToolDispatcher("read")
    elif request_class == "mutating_tool":
        tool_dispatcher = _BenchToolDispatcher("mutate")
    assistant._tool_dispatcher = tool_dispatcher
    assistant._ha_bridge = None
    assistant._shopping_list_handler = None
    assistant._music_handler = None
    assistant._device_state_handler = None
    assistant._skill_trainer = None
    assistant._skill_router = None
    assistant._action_dispatcher._tool_dispatcher = tool_dispatcher
    assistant._action_dispatcher._ha_bridge = None
    assistant._action_dispatcher._shopping_list_handler = None
    assistant._action_dispatcher._music_handler = None
    assistant._action_dispatcher._device_state_handler = None
    assistant._action_dispatcher._skill_trainer = None
    assistant._action_dispatcher._skill_router = None
    return assistant, startup_ms


def _chat_sample(
    request_class: str, *, warm_state: str, assistant=None
) -> tuple[BenchmarkSample, object]:
    startup_ms = 0.0
    if assistant is None:
        assistant, startup_ms = _make_assistant(request_class)

    logger = logging.getLogger("rex.assistant")
    capture = _RecordCapture("chat_latency")
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(capture)
    try:
        asyncio.run(
            assistant.generate_reply(f"benchmark {request_class}", active_user_id="benchmark")
        )
    finally:
        logger.removeHandler(capture)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate
    if not capture.records:
        raise RuntimeError("Chat latency event was not emitted")
    record = capture.records[-1]
    stages = {
        key[:-3]: float(value)
        for key, value in vars(record).items()
        if key.endswith("_ms") and isinstance(value, (int, float))
    }
    if warm_state == "cold":
        stages["startup"] = startup_ms
        stages["total"] = stages.get("total", 0.0) + startup_ms
    return (
        BenchmarkSample(
            request_class=request_class,
            warm_state=warm_state,
            evidence_class="deterministic_mock",
            stages_ms=stages,
        ),
        assistant,
    )


def _stream_chat_sample(*, warm_state: str, assistant=None) -> tuple[BenchmarkSample, object]:
    startup_ms = 0.0
    if assistant is None:
        assistant, startup_ms = _make_assistant("typed_chat")

    logger = logging.getLogger("rex.assistant")
    capture = _RecordCapture("chat_stream_latency")
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(capture)

    async def _consume() -> None:
        async for _chunk in assistant.stream_reply(
            "benchmark typed_chat", active_user_id="benchmark"
        ):
            pass

    try:
        asyncio.run(_consume())
    finally:
        logger.removeHandler(capture)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate
    if not capture.records:
        raise RuntimeError("Streaming chat latency event was not emitted")
    record = capture.records[-1]
    stages = {
        key[:-3]: float(value)
        for key, value in vars(record).items()
        if key.endswith("_ms") and isinstance(value, (int, float))
    }
    if warm_state == "cold":
        stages["startup"] = startup_ms
        stages["total"] = stages.get("total", 0.0) + startup_ms
    return (
        BenchmarkSample(
            request_class="typed_chat",
            warm_state=warm_state,
            evidence_class="deterministic_mock",
            stages_ms=stages,
        ),
        assistant,
    )


class _OnceListener:
    async def listen(self, _source):
        yield b"wake"

    def mark_listening_started(self, *, reason: str = "benchmark") -> None:
        return None

    def reset(self, *, reason: str = "benchmark") -> None:
        return None


def _voice_sample(*, warm_state: str) -> BenchmarkSample:
    async def detection_source():
        return b"wake"

    async def record_phrase():
        time.sleep(0.001)
        return b"audio"

    async def transcribe(_audio):
        await asyncio.sleep(0.001)
        return "benchmark voice request"

    async def speak(_text: str) -> None:
        await asyncio.sleep(0.001)

    assistant = MagicMock()
    assistant.generate_reply = AsyncMock(return_value="benchmark reply")
    del assistant.stream_reply

    started = time.perf_counter_ns()
    loop = VoiceLoop(
        assistant,
        wake_listener=_OnceListener(),
        detection_source=detection_source,
        record_phrase=record_phrase,
        transcribe=transcribe,
        speak=speak,
        post_interaction_cooldown=0,
        post_wake_preroll_seconds=0,
    )
    startup_ms = (time.perf_counter_ns() - started) / 1_000_000

    logger = logging.getLogger("rex.voice_loop")
    capture = logging.Handler()
    records: list[logging.LogRecord] = []
    capture.emit = lambda record: records.append(record)  # type: ignore[method-assign]
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(capture)
    try:
        asyncio.run(loop.run(max_interactions=1))
    finally:
        logger.removeHandler(capture)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate

    by_event = {
        record.event: record
        for record in records
        if getattr(record, "event", None)
        in {
            "wake_detected",
            "capture_started",
            "capture_ended",
            "stt_completed",
            "llm_completed",
            "playback_completed",
        }
    }
    required = {
        "wake_detected",
        "capture_started",
        "capture_ended",
        "stt_completed",
        "llm_completed",
        "playback_completed",
    }
    if set(by_event) != required:
        raise RuntimeError(f"Voice latency events missing: {sorted(required - set(by_event))}")
    wake = by_event["wake_detected"]
    playback = by_event["playback_completed"]
    total_ms = ((playback.start_ns - wake.start_ns) / 1_000_000) + float(playback.duration_ms)
    stages = {
        "capture": float(by_event["capture_ended"].duration_ms),
        "stt": float(by_event["stt_completed"].duration_ms),
        "llm": float(by_event["llm_completed"].duration_ms),
        "first_audio": (playback.start_ns - wake.start_ns) / 1_000_000,
        "completion": total_ms,
        "total": total_ms,
    }
    if warm_state == "cold":
        stages["startup"] = startup_ms
        stages["total"] += startup_ms
    return BenchmarkSample(
        request_class="voice",
        warm_state=warm_state,
        evidence_class="deterministic_mock",
        stages_ms=stages,
    )


def run_baseline(iterations: int) -> dict:
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    samples: list[BenchmarkSample] = []
    for request_class in REQUEST_CLASSES:
        if request_class == "voice":
            for _ in range(iterations):
                samples.append(_voice_sample(warm_state="cold"))
                samples.append(_voice_sample(warm_state="warm"))
            continue
        if request_class == "typed_chat":
            warm_assistant, _ = _make_assistant(request_class)
            for _ in range(iterations):
                cold, _ = _stream_chat_sample(warm_state="cold")
                warm, warm_assistant = _stream_chat_sample(
                    warm_state="warm", assistant=warm_assistant
                )
                samples.extend((cold, warm))
            continue
        warm_assistant, _ = _make_assistant(request_class)
        for _ in range(iterations):
            cold, _ = _chat_sample(request_class, warm_state="cold")
            warm, warm_assistant = _chat_sample(
                request_class, warm_state="warm", assistant=warm_assistant
            )
            samples.extend((cold, warm))
    return build_report(samples, profile="baseline")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("baseline",), default="baseline")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = run_baseline(args.iterations)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(f"RexBench baseline written: {args.output}")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
