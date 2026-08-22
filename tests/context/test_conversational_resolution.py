from __future__ import annotations

import asyncio

from rex.context.active import ActiveContextRef, ActiveContextStore
from rex.mobile_api.action_context import mobile_action_context
from rex.tools.protocol import ToolResult
from tests.test_us016_action_dispatcher import (
    _make_context,
    _make_dispatcher,
    _unhandled_intent,
)


def _put(
    store: ActiveContextStore,
    *,
    domain: str,
    key: str,
    payload: dict[str, object],
) -> None:
    store.put(
        ActiveContextRef(
            domain=domain,
            key=key,
            owner_user_id="james",
            payload=payload,
            source_ids=(),
            revision="local:1",
            expires_at=200.0,
        )
    )


def test_media_language_narrows_other_active_domains():
    store = ActiveContextStore(clock=lambda: 100.0)
    _put(
        store,
        domain="media",
        key="session-1",
        payload={"target_id": "ha:media_player.living_room"},
    )
    _put(store, domain="timekeeping", key="timer-1", payload={"record_type": "timer"})

    result = store.resolve(
        "james",
        "pause the music",
        candidate_domains=("media", "timekeeping"),
    )

    assert result.ref is not None
    assert result.ref.domain == "media"
    assert result.reason == "resolved"


def test_timer_language_narrows_media_reference():
    store = ActiveContextStore(clock=lambda: 100.0)
    _put(store, domain="media", key="session-1", payload={"target_id": "speaker"})
    _put(store, domain="timekeeping", key="timer-1", payload={"record_type": "timer"})

    result = store.resolve(
        "james",
        "cancel the timer",
        candidate_domains=("media", "timekeeping"),
    )

    assert result.ref is not None
    assert result.ref.domain == "timekeeping"
    assert result.ref.key == "timer-1"


def test_explicit_reference_key_selects_one_of_same_domain():
    store = ActiveContextStore(clock=lambda: 100.0)
    _put(store, domain="timekeeping", key="timer-1", payload={"record_type": "timer"})
    _put(store, domain="timekeeping", key="timer-2", payload={"record_type": "timer"})

    result = store.resolve(
        "james",
        "cancel timer-2",
        candidate_domains=("timekeeping",),
    )

    assert result.ref is not None
    assert result.ref.key == "timer-2"
    assert result.reason == "resolved"


def test_unrelated_utterance_does_not_guess_reference():
    store = ActiveContextStore(clock=lambda: 100.0)
    _put(store, domain="media", key="session-1", payload={"target_id": "speaker"})

    result = store.resolve(
        "james",
        "what is the weather tomorrow",
        candidate_domains=("media", "timekeeping"),
    )

    assert result.ref is None
    assert result.reason == "not_referential"


class _RecordingToolDispatcher:
    def __init__(self) -> None:
        self.dispatch_calls: list[tuple[str, dict, dict]] = []

    def dispatch(self, name, args, context):
        self.dispatch_calls.append((name, args, context))
        return ToolResult(
            success=True,
            output={"tool": name, "args": args},
            status="verified" if name.endswith("manage") else "completed",
        )

    def select_tools(self, transcript):
        return []

    def format_tool_context(self, results):
        return f"active={results!r}"


def test_dispatcher_resolves_cancel_it_to_single_active_timer():
    store = ActiveContextStore(clock=lambda: 100.0)
    _put(store, domain="timekeeping", key="timer-1", payload={"record_type": "timer"})
    tools = _RecordingToolDispatcher()
    dispatcher = _make_dispatcher(tool_dispatcher=tools, active_context_store=store)
    asyncio.run(
        dispatcher.dispatch(
            _unhandled_intent(),
            _make_context(),
            "cancel it",
            user_id="james",
        )
    )

    assert tools.dispatch_calls == [
        (
            "timekeeping_manage",
            {"action": "cancel_timer", "reference": "timer-1"},
            {"user_id": "james"},
        )
    ]


def test_dispatcher_resolves_pause_it_to_single_active_media_target():
    store = ActiveContextStore(clock=lambda: 100.0)
    _put(
        store,
        domain="media",
        key="ha:media_player.living_room",
        payload={"target_id": "ha:media_player.living_room", "provider": "ha"},
    )
    tools = _RecordingToolDispatcher()
    dispatcher = _make_dispatcher(tool_dispatcher=tools, active_context_store=store)
    asyncio.run(
        dispatcher.dispatch(
            _unhandled_intent(),
            _make_context(),
            "pause it",
            user_id="james",
        )
    )

    assert len(tools.dispatch_calls) == 1
    name, args, context = tools.dispatch_calls[0]
    assert name == "media_manage"
    assert args["action"] == "pause"
    assert args["target_text"] == "ha:media_player.living_room"
    assert context == {"user_id": "james"}


def test_dispatcher_clarifies_two_active_timers_instead_of_guessing():
    store = ActiveContextStore(clock=lambda: 100.0)
    _put(store, domain="timekeeping", key="timer-1", payload={"record_type": "timer"})
    _put(store, domain="timekeeping", key="timer-2", payload={"record_type": "timer"})
    tools = _RecordingToolDispatcher()
    dispatcher = _make_dispatcher(tool_dispatcher=tools, active_context_store=store)

    result = asyncio.run(
        dispatcher.dispatch(
            _unhandled_intent(),
            _make_context(),
            "cancel it",
            user_id="james",
        )
    )

    assert result.success is True
    assert result.response == "Which timer do you mean?"
    assert result.actions_taken == ["context_clarification"]
    assert tools.dispatch_calls == []


def test_mobile_active_mutation_reference_does_not_bypass_structured_auth():
    store = ActiveContextStore(clock=lambda: 100.0)
    _put(store, domain="timekeeping", key="timer-1", payload={"record_type": "timer"})
    tools = _RecordingToolDispatcher()
    dispatcher = _make_dispatcher(tool_dispatcher=tools, active_context_store=store)

    with mobile_action_context(frozenset({"chat.send"})):
        result = asyncio.run(
            dispatcher.dispatch(
                _unhandled_intent(),
                _make_context(),
                "cancel it",
                user_id="james",
            )
        )

    assert result.response == "LLM reply"
    assert tools.dispatch_calls == []
