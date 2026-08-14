from __future__ import annotations

import asyncio
import inspect
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
    assistant._context_builder = MagicMock()
    assistant._context_builder.build.return_value = SimpleNamespace(
        messages=[], prompt="prompt", system_prompt="system"
    )
    assistant._response_builder = MagicMock()
    assistant._response_builder.check_cache.return_value = None
    assistant._turn_events: list = []
    assistant._turn_event_observer = assistant._turn_events.append
    return assistant


def _unhandled_intent(assistant: Assistant) -> None:
    router = MagicMock()
    router.route.return_value = IntentResult(handled=False, response=None, intent_type=None)
    assistant._intent_router = router


def _final_response(text: str) -> FinalResponse:
    return FinalResponse(text=text, tts_text=text)


def test_direct_answer_runs_inside_turn_engine() -> None:
    assistant = _assistant()
    router = MagicMock()
    router.route.return_value = IntentResult(
        handled=True,
        response="Hello. How can I help?",
        intent_type="greeting",
    )
    assistant._intent_router = router

    reply = asyncio.run(assistant.generate_reply("hello"))

    assert reply == "Hello. How can I help?"
    assert assistant._turn_events[0].kind is EventKind.TURN_STARTED
    assert assistant._turn_events[-1].kind is EventKind.COMPLETED
    assert any(event.kind is EventKind.ROUTE_PROGRESS for event in assistant._turn_events)
    assert any(event.kind is EventKind.RESPONSE_PROGRESS for event in assistant._turn_events)


def test_cache_hit_does_not_bypass_turn_engine() -> None:
    assistant = _assistant()
    _unhandled_intent(assistant)
    assistant._response_builder.check_cache.return_value = "cached-answer"
    assistant._action_dispatcher = MagicMock()

    reply = asyncio.run(assistant.generate_reply("cached question"))

    assert reply == "cached-answer"
    assistant._action_dispatcher.dispatch.assert_not_called()
    assert assistant._turn_events[0].kind is EventKind.TURN_STARTED
    assert assistant._turn_events[-1].kind is EventKind.COMPLETED


def test_read_only_tool_result_preserves_response_and_event_stream() -> None:
    assistant = _assistant()
    _unhandled_intent(assistant)
    seen_streams: list[TurnEventStream] = []

    async def dispatch(*_args, turn_events=None, **_kwargs):
        assert isinstance(turn_events, TurnEventStream)
        seen_streams.append(turn_events)
        turn_events.emit(
            EventKind.CAPABILITY_PROGRESS,
            {"capability": "web_search", "operation": "read"},
        )
        turn_events.emit(EventKind.ACTION_PROGRESS, {"status": "completed"})
        return ActionResult(success=True, response="tool-answer", actions_taken=["web_search"])

    assistant._action_dispatcher = MagicMock()
    assistant._action_dispatcher.dispatch = dispatch
    assistant._response_builder.build.return_value = _final_response("tool-answer")

    reply = asyncio.run(assistant.generate_reply("search the web"))

    assert reply == "tool-answer"
    assert seen_streams
    assert any(event.kind is EventKind.CAPABILITY_PROGRESS for event in assistant._turn_events)
    assert any(event.kind is EventKind.ACTION_PROGRESS for event in assistant._turn_events)


def test_mutation_confirmation_wording_is_preserved() -> None:
    assistant = _assistant()
    _unhandled_intent(assistant)

    async def dispatch(*_args, turn_events=None, **_kwargs):
        assert isinstance(turn_events, TurnEventStream)
        turn_events.emit(
            EventKind.CAPABILITY_PROGRESS,
            {"capability": "home_assistant", "operation": "mutate"},
        )
        turn_events.emit(
            EventKind.ACTION_PROGRESS,
            {"status": "confirmation_required"},
        )
        return ActionResult(
            success=False,
            response="Please confirm locking the front door.",
            actions_taken=["home_assistant"],
        )

    assistant._action_dispatcher = MagicMock()
    assistant._action_dispatcher.dispatch = dispatch
    assistant._response_builder.build.return_value = _final_response(
        "Please confirm locking the front door."
    )

    reply = asyncio.run(assistant.generate_reply("lock the front door"))

    assert reply == "Please confirm locking the front door."
    assert assistant._turn_events[-1].kind is EventKind.COMPLETED


def test_model_failure_is_re_raised_with_failed_terminal() -> None:
    assistant = _assistant()
    _unhandled_intent(assistant)

    async def dispatch(*_args, turn_events=None, **_kwargs):
        assert isinstance(turn_events, TurnEventStream)
        turn_events.emit(
            EventKind.MODEL_PROGRESS,
            {"stage": "generation", "status": "started"},
        )
        raise RuntimeError("model unavailable")

    assistant._action_dispatcher = MagicMock()
    assistant._action_dispatcher.dispatch = dispatch

    with pytest.raises(RuntimeError, match="model unavailable"):
        asyncio.run(assistant.generate_reply("explain this"))

    assert assistant._turn_events[-1].kind is EventKind.FAILED
    assert sum(event.is_terminal for event in assistant._turn_events) == 1


def test_unavailable_capability_response_shape_is_preserved() -> None:
    assistant = _assistant()
    _unhandled_intent(assistant)

    async def dispatch(*_args, turn_events=None, **_kwargs):
        assert isinstance(turn_events, TurnEventStream)
        turn_events.emit(
            EventKind.CAPABILITY_PROGRESS,
            {"capability": "calendar_write", "status": "unavailable"},
        )
        return ActionResult(
            success=False,
            response="Calendar write is unavailable.",
            error="unavailable",
        )

    assistant._action_dispatcher = MagicMock()
    assistant._action_dispatcher.dispatch = dispatch
    assistant._response_builder.build.return_value = _final_response(
        "Calendar write is unavailable."
    )

    reply = asyncio.run(assistant.generate_reply("add an event"))

    assert reply == "Calendar write is unavailable."
    assert assistant._turn_events[-1].kind is EventKind.COMPLETED


def test_missing_identity_fails_before_turn_engine() -> None:
    assistant = _assistant(user_id=None)
    _unhandled_intent(assistant)
    engine = MagicMock()
    assistant._turn_engine = engine

    with pytest.raises(IdentityRequiredError):
        asyncio.run(assistant.generate_reply("hello"))

    engine.execute_async.assert_not_called()
    assert assistant._turn_events == []


def test_real_dispatcher_emits_tool_model_and_result_boundaries() -> None:
    from rex.actions.dispatcher import ActionDispatcher
    from rex.runtime.turn import (
        AuthorizationSnapshotRef,
        ResponseMode,
        TurnContext,
        TurnScope,
        TurnSource,
    )

    context_builder = MagicMock()
    context_builder.build.return_value = SimpleNamespace(messages=[], prompt="prompt")
    llm = MagicMock()
    llm.generate.return_value = "model-answer"
    result_handler = MagicMock()

    async def process(*_args, **_kwargs):
        return "model-answer"

    result_handler.process = process
    tool = SimpleNamespace(name="time_now", operation="read")
    tool_dispatcher = MagicMock()
    tool_dispatcher.select_tools.return_value = [tool]
    tool_dispatcher.execute_tools.return_value = [{"ok": True}]
    tool_dispatcher.format_tool_context.return_value = "time context"

    dispatcher = ActionDispatcher(
        context_builder=context_builder,
        llm=llm,
        result_handler=result_handler,
        tool_dispatcher=tool_dispatcher,
    )
    turn_context = TurnContext.create(
        user_id="james",
        scope=TurnScope.USER,
        source=TurnSource.ASSISTANT,
        device_id=None,
        response_mode=ResponseMode.SCREEN,
        authorization=AuthorizationSnapshotRef(
            "rex-policy:existing-runtime",
            "rex-permissions:validated-user:james",
        ),
    )
    observed = []
    stream = TurnEventStream(turn_context, observer=observed.append)

    result = asyncio.run(
        dispatcher.dispatch(
            IntentResult(handled=False, response=None, intent_type=None),
            context_builder.build.return_value,
            "what time is it",
            user_id="james",
            turn_events=stream,
        )
    )

    assert result.response == "model-answer"
    assert any(
        event.kind is EventKind.CAPABILITY_PROGRESS
        and "time_now" in event.details.get("capabilities", [])
        for event in observed
    )
    assert any(
        event.kind is EventKind.ACTION_PROGRESS and event.details.get("stage") == "tool_execution"
        for event in observed
    )
    model_events = [event for event in observed if event.kind is EventKind.MODEL_PROGRESS]
    assert [event.details.get("status") for event in model_events] == [
        "started",
        "returned",
    ]
    assert any(
        event.kind is EventKind.ACTION_PROGRESS and event.details.get("stage") == "result_handler"
        for event in observed
    )


def test_generate_reply_has_no_direct_model_shortcut() -> None:
    source = inspect.getsource(Assistant.generate_reply)

    assert "execute_async" in source
    assert "._llm.generate" not in source
    assert "._generate_model_reply" not in source


def test_voice_mode_does_not_invent_surface_provenance() -> None:
    from rex.runtime.turn import ResponseMode, TurnSource

    context = _assistant()._build_turn_context("james", voice_mode=True)

    assert context.source is TurnSource.ASSISTANT
    assert context.response_mode is ResponseMode.VOICE


def test_canonical_turn_passes_authority_and_selected_model_to_context_cache() -> None:
    from rex.runtime.turn import TurnScope

    assistant = _assistant()
    assistant._llm.provider = "test-provider"
    assistant._llm.model_name = "selected-model"
    _unhandled_intent(assistant)

    async def dispatch(*_args, **_kwargs):
        return ActionResult(success=True, response="model-answer")

    assistant._action_dispatcher = MagicMock()
    assistant._action_dispatcher.dispatch = dispatch
    assistant._response_builder.build.return_value = _final_response("model-answer")

    reply = asyncio.run(assistant.generate_reply("explain this"))

    assert reply == "model-answer"
    cache_request = assistant._context_builder.build.call_args.kwargs["cache_request"]
    assert cache_request.user_id == "james"
    assert cache_request.scope is TurnScope.USER
    assert cache_request.authorization.policy_ref == "rex-policy:existing-runtime"
    assert cache_request.authorization.permission_ref == "rex-permissions:validated-user:james"
    assert cache_request.model_provider == "test-provider"
    assert cache_request.model_name == "selected-model"
