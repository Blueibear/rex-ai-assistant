from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from rex.assistant import Assistant
from rex.intent.router import IntentResult
from rex.runtime.events import EventKind, TurnEventStream
from rex.runtime.status import TurnStatus, TurnStatusProjector, project_turn_status
from rex.runtime.turn import (
    AuthorizationSnapshotRef,
    ResponseMode,
    TurnContext,
    TurnScope,
    TurnSource,
)


def _context(user_id: str = "james") -> TurnContext:
    return TurnContext.create(
        user_id=user_id,
        scope=TurnScope.USER,
        source=TurnSource.CLI,
        device_id=None,
        response_mode=ResponseMode.SCREEN,
        authorization=AuthorizationSnapshotRef("policy:test", f"permissions:{user_id}"),
    )


def _assistant() -> Assistant:
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
    assistant._user_id = "james"
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
    assistant._log_turn = MagicMock()
    router = MagicMock()
    router.route.return_value = IntentResult(
        handled=True,
        response="Direct answer.",
        intent_type="direct",
    )
    assistant._intent_router = router
    assistant._turn_events = []
    assistant._turn_event_observer = assistant._turn_events.append
    return assistant


def test_projector_maps_canonical_events_to_progressive_statuses() -> None:
    stream = TurnEventStream(_context())
    events = [
        stream.emit(EventKind.TURN_STARTED),
        stream.emit(EventKind.ROUTE_PROGRESS, {"stage": "intent"}),
        stream.emit(EventKind.CONTEXT_PROGRESS, {"stage": "built"}),
        stream.emit(EventKind.CAPABILITY_PROGRESS, {"stage": "tool_selection"}),
        stream.emit(EventKind.MODEL_PROGRESS, {"status": "started"}),
        stream.emit(EventKind.ACTION_PROGRESS, {"stage": "tool_execution", "status": "returned"}),
        stream.emit(EventKind.ACTION_PROGRESS, {"stage": "verification", "status": "verified"}),
        stream.emit(EventKind.RESPONSE_PROGRESS, {"stage": "delta"}),
        stream.finish(EventKind.COMPLETED),
    ]

    assert [project_turn_status(event).status for event in events] == [
        TurnStatus.THINKING,
        TurnStatus.CHECKING,
        TurnStatus.CHECKING,
        TurnStatus.CHECKING,
        TurnStatus.THINKING,
        TurnStatus.ACTING,
        TurnStatus.VERIFYING,
        TurnStatus.SPEAKING,
        TurnStatus.DONE,
    ]
    assert project_turn_status(events[-1]).terminal is True


def test_projected_payload_is_privacy_safe_even_when_event_details_are_not() -> None:
    private_marker = "garage-code-4242"
    stream = TurnEventStream(_context())
    event = stream.emit(
        EventKind.ACTION_PROGRESS,
        {
            "stage": "tool_execution",
            "transcript": private_marker,
            "prompt": private_marker,
            "memory": private_marker,
            "credential": private_marker,
            "tool_result": private_marker,
        },
    )

    update = project_turn_status(event)
    payload = update.to_dict()

    assert set(payload) == {"turn_id", "sequence", "status", "terminal"}
    assert private_marker not in repr(payload)
    assert "james" not in repr(payload)


def test_terminal_failure_and_cancellation_clear_nonterminal_status() -> None:
    failed_stream = TurnEventStream(_context())
    failed = failed_stream.finish(EventKind.FAILED, {"reason": "private failure text"})
    cancelled_stream = TurnEventStream(_context())
    cancelled = cancelled_stream.finish(EventKind.CANCELLED, {"reason": "user transcript"})

    assert project_turn_status(failed).status is TurnStatus.ERROR
    assert project_turn_status(failed).terminal is True
    assert project_turn_status(cancelled).status is TurnStatus.CANCELLED
    assert project_turn_status(cancelled).terminal is True


def test_status_projector_deduplicates_presentation_only_not_terminal_truth() -> None:
    updates = []
    projector = TurnStatusProjector(updates.append)
    stream = TurnEventStream(_context())

    projector.observe(stream.emit(EventKind.ROUTE_PROGRESS))
    projector.observe(stream.emit(EventKind.CONTEXT_PROGRESS))
    projector.observe(stream.emit(EventKind.CAPABILITY_PROGRESS))
    projector.observe(stream.finish(EventKind.COMPLETED))

    assert [item.status for item in updates] == [TurnStatus.CHECKING, TurnStatus.DONE]
    assert updates[-1].terminal is True


def test_explicit_turn_observers_are_isolated_per_request_and_legacy_observer_still_runs() -> None:
    assistant = _assistant()
    james_events = []
    cole_events = []

    async def run_both() -> None:
        await asyncio.gather(
            assistant.generate_reply(
                "hello",
                active_user_id="james",
                event_observer=james_events.append,
            ),
            assistant.generate_reply(
                "hello",
                active_user_id="cole",
                event_observer=cole_events.append,
            ),
        )

    asyncio.run(run_both())

    assert james_events and cole_events
    assert {event.user_id for event in james_events} == {"james"}
    assert {event.user_id for event in cole_events} == {"cole"}
    assert len(assistant._turn_events) == len(james_events) + len(cole_events)
    assert all(events[-1].kind is EventKind.COMPLETED for events in (james_events, cole_events))


def test_mobile_chat_service_projects_real_turn_events_without_private_details() -> None:
    from rex.mobile_api.chat import MobileChatService

    class EventfulAssistant:
        async def stream_reply(self, _message, *, active_user_id=None, event_observer=None):
            stream = TurnEventStream(_context(active_user_id or "james"), observer=event_observer)
            stream.emit(EventKind.TURN_STARTED, {"prompt": "private prompt"})
            stream.emit(
                EventKind.ACTION_PROGRESS,
                {"stage": "tool_execution", "tool_result": "private_marker"},
            )
            yield "answer"
            stream.finish(EventKind.COMPLETED, {"memory": "private memory"})

    service = MobileChatService(lambda: EventfulAssistant())
    updates = []

    chunks = list(service.stream("hello", user_id="james", status_observer=updates.append))

    assert chunks == ["answer"]
    assert [update.status for update in updates] == [
        TurnStatus.THINKING,
        TurnStatus.ACTING,
        TurnStatus.DONE,
    ]
    assert all(
        set(update.to_dict()) == {"turn_id", "sequence", "status", "terminal"} for update in updates
    )
    assert "private" not in repr([update.to_dict() for update in updates])


def test_mobile_chat_service_preserves_legacy_assistant_signature_with_status_sink() -> None:
    from rex.mobile_api.chat import MobileChatService

    class LegacyAssistant:
        async def generate_reply(self, _message, *, voice_mode=False, active_user_id=None):
            return f"reply:{active_user_id}:{voice_mode}"

        async def stream_reply(self, _message, *, active_user_id=None):
            yield f"stream:{active_user_id}"

    service = MobileChatService(lambda: LegacyAssistant())
    updates = []

    assert (
        service.generate("hello", user_id="james", status_observer=updates.append)
        == "reply:james:False"
    )
    assert list(service.stream("hello", user_id="james", status_observer=updates.append)) == [
        "stream:james"
    ]
    assert updates == []


def test_all_user_interfaces_depend_on_the_canonical_status_projector() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    sources = {
        "cli": (root / "rex" / "commands" / "core.py").read_text(encoding="utf-8"),
        "electron": (root / "bridge" / "rex_chat_stream_bridge.py").read_text(encoding="utf-8"),
        "voice": (root / "bridge" / "rex_voice_bridge.py").read_text(encoding="utf-8"),
        "mobile": (root / "rex" / "mobile_api" / "chat.py").read_text(encoding="utf-8"),
    }

    for name, source in sources.items():
        assert "TurnStatusProjector" in source, f"{name} bypasses canonical progressive status"
    assert 'status in {"thinking", "executing"}' not in sources["voice"]


def test_stream_terminal_status_follows_final_delivered_chunk() -> None:
    assistant = _assistant()
    assistant._intent_router.route.return_value = IntentResult(
        handled=True,
        response="First sentence. Second sentence.",
        intent_type="direct",
    )
    observed: list[tuple[str, str]] = []
    projector = TurnStatusProjector(lambda update: observed.append(("status", update.status.value)))

    async def consume() -> None:
        async for chunk in assistant.stream_reply(
            "hello",
            active_user_id="james",
            event_observer=projector.observe,
        ):
            observed.append(("chunk", chunk))
            if chunk.startswith("First sentence"):
                await asyncio.sleep(0.05)

    asyncio.run(consume())

    terminal_index = observed.index(("status", "done"))
    chunk_indexes = [index for index, item in enumerate(observed) if item[0] == "chunk"]
    assert chunk_indexes
    assert max(chunk_indexes) < terminal_index, observed
