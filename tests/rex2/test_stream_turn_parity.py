from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rex.actions.dispatcher import ActionResult
from rex.assistant import Assistant
from rex.assistant_errors import IdentityRequiredError
from rex.intent.router import IntentResult
from rex.response.builder import FinalResponse
from rex.runtime.events import EventKind, TurnEventStream


def _assistant(*, user_id: str | None = "james") -> Assistant:
    assistant = Assistant.__new__(Assistant)
    assistant._settings = SimpleNamespace(
        max_memory_items=50,
        persist_history=False,
        followups_enabled=False,
        model_routing=None,
        transcripts_enabled=False,
        llm_provider="test",
        llm_model="test-model",
        llm=None,
    )
    assistant._user_id = user_id
    assistant._histories = {}
    assistant._history_limit = 50
    assistant._plugins = []
    assistant._history_store = None
    assistant._followup_engine = None
    assistant._followup_sessions = set()
    assistant._followup_bootstrap_pending = False
    assistant._pending_followups = {}
    assistant._router = None
    assistant._response_cache = None
    assistant._ha_bridge = None
    assistant._suggestion_engine = None
    assistant._pattern_entries = {}
    assistant._llm = MagicMock()
    assistant._llm.model_name = "test-model"
    assistant._llm.stream.return_value = iter(["legacy-stream"])
    assistant._context_builder = MagicMock()
    assistant._context_builder.build.return_value = SimpleNamespace(
        messages=[], prompt="prompt", system_prompt="system"
    )
    assistant._response_builder = MagicMock()
    assistant._response_builder.check_cache.return_value = None
    assistant._response_builder.build.side_effect = lambda result, _ctx, **_kwargs: FinalResponse(
        text=result.response,
        tts_text=result.response,
    )
    assistant._turn_events: list = []
    assistant._turn_event_observer = assistant._turn_events.append
    return assistant


def _unhandled(assistant: Assistant) -> None:
    router = MagicMock()
    router.route.return_value = IntentResult(handled=False, response=None, intent_type=None)
    assistant._intent_router = router


async def _collect(assistant: Assistant, text: str, **kwargs) -> list[str]:
    return [chunk async for chunk in assistant.stream_reply(text, **kwargs)]


def _join(chunks: list[str]) -> str:
    return " ".join(chunk.strip() for chunk in chunks if chunk.strip()).strip()


def test_stream_direct_answer_uses_turn_engine_and_matches_generate_reply() -> None:
    assistant = _assistant()
    router = MagicMock()
    router.route.return_value = IntentResult(
        handled=True,
        response="Hello. How can I help?",
        intent_type="greeting",
    )
    assistant._intent_router = router

    generated = asyncio.run(assistant.generate_reply("hello"))
    assistant._turn_events.clear()
    streamed = asyncio.run(_collect(assistant, "hello"))

    assert _join(streamed) == generated
    assert assistant._turn_events[0].kind is EventKind.TURN_STARTED
    assert assistant._turn_events[-1].kind is EventKind.COMPLETED
    assert sum(event.is_terminal for event in assistant._turn_events) == 1


def test_stream_cache_hit_matches_generate_and_skips_dispatch() -> None:
    assistant = _assistant()
    _unhandled(assistant)
    assistant._response_builder.check_cache.return_value = "Cached answer."
    assistant._action_dispatcher = MagicMock()

    generated = asyncio.run(assistant.generate_reply("same question"))
    assistant._turn_events.clear()
    streamed = asyncio.run(_collect(assistant, "same question"))

    assert _join(streamed) == generated == "Cached answer."
    assistant._action_dispatcher.dispatch.assert_not_called()
    assert any(
        event.kind is EventKind.ROUTE_PROGRESS
        and event.details.get("stage") == "cache"
        and event.details.get("cache_hit") is True
        for event in assistant._turn_events
    )


@pytest.mark.parametrize(
    ("response", "success", "capability", "status"),
    [
        ("The current result is 72 degrees.", True, "weather", "returned"),
        (
            "Please confirm locking the front door.",
            False,
            "home_assistant",
            "confirmation_required",
        ),
        ("Calendar write is unavailable.", False, "calendar_write", "unavailable"),
    ],
)
def test_stream_action_outcomes_match_nonstreaming(
    response: str, success: bool, capability: str, status: str
) -> None:
    assistant = _assistant()
    _unhandled(assistant)

    async def dispatch(*_args, turn_events=None, **_kwargs):
        assert isinstance(turn_events, TurnEventStream)
        turn_events.emit(
            EventKind.CAPABILITY_PROGRESS,
            {"capability": capability, "status": "selected"},
        )
        turn_events.emit(
            EventKind.ACTION_PROGRESS,
            {"capability": capability, "status": status},
        )
        return ActionResult(success=success, response=response, actions_taken=[capability])

    assistant._action_dispatcher = MagicMock()
    assistant._action_dispatcher.dispatch = dispatch

    generated = asyncio.run(assistant.generate_reply("do the thing"))
    assistant._turn_events.clear()
    streamed = asyncio.run(_collect(assistant, "do the thing"))

    assert _join(streamed) == generated == response
    assert any(
        event.kind is EventKind.ACTION_PROGRESS and event.details.get("status") == status
        for event in assistant._turn_events
    )
    assert assistant._turn_events[-1].kind is EventKind.COMPLETED


def test_stream_model_failure_matches_generate_failure_and_terminal() -> None:
    assistant = _assistant()
    _unhandled(assistant)

    async def dispatch(*_args, turn_events=None, **_kwargs):
        assert isinstance(turn_events, TurnEventStream)
        turn_events.emit(EventKind.MODEL_PROGRESS, {"status": "started"})
        raise RuntimeError("provider failed")

    assistant._action_dispatcher = MagicMock()
    assistant._action_dispatcher.dispatch = dispatch

    with pytest.raises(RuntimeError, match="provider failed"):
        asyncio.run(assistant.generate_reply("fail"))

    assistant._turn_events.clear()
    with pytest.raises(RuntimeError, match="provider failed"):
        asyncio.run(_collect(assistant, "fail"))

    assert assistant._turn_events[-1].kind is EventKind.FAILED
    assert sum(event.is_terminal for event in assistant._turn_events) == 1


def test_stream_fails_closed_before_turn_engine_without_identity() -> None:
    assistant = _assistant(user_id=None)
    _unhandled(assistant)
    engine = MagicMock()
    assistant._turn_engine = engine

    with pytest.raises(IdentityRequiredError):
        asyncio.run(_collect(assistant, "hello"))

    engine.execute_async.assert_not_called()
    assert assistant._turn_events == []


def test_stream_response_events_are_ordered_before_terminal() -> None:
    assistant = _assistant()
    _unhandled(assistant)

    async def dispatch(*_args, **_kwargs):
        return ActionResult(
            success=True,
            response="First sentence. Second sentence.",
            actions_taken=["llm"],
        )

    assistant._action_dispatcher = MagicMock()
    assistant._action_dispatcher.dispatch = dispatch

    chunks = asyncio.run(_collect(assistant, "two sentences"))

    assert chunks == ["First sentence.", "Second sentence."]
    response_events = [
        event
        for event in assistant._turn_events
        if event.kind is EventKind.RESPONSE_PROGRESS and event.details.get("stage") == "delta"
    ]
    assert [event.details.get("index") for event in response_events] == [0, 1]
    assert all(event.sequence < assistant._turn_events[-1].sequence for event in response_events)
    assert assistant._turn_events[-1].kind is EventKind.COMPLETED


@pytest.mark.parametrize(
    ("resolved_model", "expected_model"),
    [("deep-model", "deep-model"), (None, "fast-model")],
)
def test_stream_model_routing_escalation_and_fallback_match_generate(
    resolved_model: str | None, expected_model: str
) -> None:
    assistant = _assistant()
    _unhandled(assistant)
    assistant._llm.model_name = "fast-model"
    router = MagicMock()
    router.classify.return_value = "complex"
    router.resolve_model.return_value = resolved_model
    assistant._router = router
    observed_models: list[str] = []

    async def dispatch(*_args, **_kwargs):
        observed_models.append(assistant._llm.model_name)
        return ActionResult(success=True, response=f"used {assistant._llm.model_name}")

    assistant._action_dispatcher = MagicMock()
    assistant._action_dispatcher.dispatch = dispatch

    generated = asyncio.run(assistant.generate_reply("complex request"))
    streamed = asyncio.run(_collect(assistant, "complex request"))

    assert generated == f"used {expected_model}"
    assert _join(streamed) == generated
    assert observed_models == [expected_model, expected_model]
    assert assistant._llm.model_name == "fast-model"
