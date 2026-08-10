from __future__ import annotations

import asyncio
import inspect

from rex.assistant import Assistant
from rex.latency import LatencyTrace
from rex.runtime.events import EventKind, TurnEventStream
from rex.runtime.turn import (
    AuthorizationSnapshotRef,
    ResponseMode,
    TurnContext,
    TurnScope,
    TurnSource,
)


def test_stream_reply_uses_shared_turn_pipeline_not_legacy_provider_stream() -> None:
    source = inspect.getsource(Assistant.stream_reply)

    assert "execute_async" in source
    assert "_run_reply_turn" in source
    assert "_stream_model_reply" not in source
    assert "_stream_home_assistant_completion" not in source
    assert "_generate_model_reply" not in source


def test_safe_stream_delivery_emits_metadata_only_sentence_deltas() -> None:
    assistant = Assistant.__new__(Assistant)
    context = TurnContext.create(
        user_id="james",
        scope=TurnScope.USER,
        source=TurnSource.ASSISTANT,
        device_id=None,
        response_mode=ResponseMode.SCREEN,
        authorization=AuthorizationSnapshotRef("policy:v1", "permissions:james:v1"),
    )
    observed = []
    stream = TurnEventStream(context, observer=observed.append)
    delivered: list[str] = []

    async def sink(chunk: str) -> None:
        delivered.append(chunk)

    asyncio.run(
        assistant._deliver_safe_response(
            "First sentence. Second sentence.",
            turn_events=stream,
            response_sink=sink,
            latency_trace=LatencyTrace(channel="chat"),
            stream_started_ns=None,
        )
    )

    assert delivered == ["First sentence.", "Second sentence."]
    assert [event.kind for event in observed] == [
        EventKind.RESPONSE_PROGRESS,
        EventKind.RESPONSE_PROGRESS,
    ]
    assert [event.details["index"] for event in observed] == [0, 1]
    assert all(event.details["stage"] == "delta" for event in observed)
    assert all("First sentence" not in repr(event.details) for event in observed)
    assert all("Second sentence" not in repr(event.details) for event in observed)
