"""US-097 interface provenance and shared-brain parity tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rex.actions.dispatcher import ActionResult
from rex.assistant import Assistant
from rex.intent.router import IntentResult
from rex.response.builder import FinalResponse
from rex.runtime.events import EventKind, TurnEventStream
from rex.runtime.turn import TurnSource


def _run(coro):
    return asyncio.run(coro)


def test_turn_invocation_defaults_and_resets() -> None:
    from rex.runtime.invocation import current_turn_invocation, turn_invocation

    assert current_turn_invocation().source is TurnSource.ASSISTANT
    assert current_turn_invocation().device_id is None
    with turn_invocation(TurnSource.ELECTRON, device_id="desktop-window"):
        current = current_turn_invocation()
        assert current.source is TurnSource.ELECTRON
        assert current.device_id == "desktop-window"
    assert current_turn_invocation().source is TurnSource.ASSISTANT
    assert current_turn_invocation().device_id is None


def test_assistant_turn_context_uses_edge_provenance() -> None:
    from rex.runtime.invocation import turn_invocation

    assistant = object.__new__(Assistant)
    with turn_invocation(TurnSource.CLI, device_id="terminal-1"):
        context = assistant._build_turn_context("james", voice_mode=False)
    assert context.source is TurnSource.CLI
    assert context.device_id == "terminal-1"


@pytest.mark.parametrize(
    "source,device_id",
    [
        (TurnSource.CLI, None),
        (TurnSource.ELECTRON, None),
        (TurnSource.VOICE, None),
        (TurnSource.MOBILE, "paired-phone"),
        (TurnSource.API, None),
        (TurnSource.TELEGRAM, None),
        (TurnSource.TELEPHONY, None),
        (TurnSource.MQTT, "room-node"),
    ],
)
def test_interface_source_does_not_change_reply_semantics(source, device_id) -> None:
    from rex.runtime.invocation import turn_invocation

    assistant = object.__new__(Assistant)
    assistant._user_id = "james"
    assistant._settings = SimpleNamespace()
    assistant._turn_event_observer = None

    async def fake_turn(_events, **_kwargs):
        return "same verified reply"

    assistant._run_reply_turn = fake_turn
    with turn_invocation(source, device_id=device_id):
        result = _run(assistant.generate_reply("same request", active_user_id="james"))
    assert result == "same verified reply"


def test_supported_interfaces_stamp_turn_source_and_do_not_call_models_directly() -> None:
    root = Path(__file__).resolve().parents[2]
    expected = {
        "rex/commands/core.py": "TurnSource.CLI",
        "bridge/rex_chat_bridge.py": "TurnSource.ELECTRON",
        "bridge/rex_chat_stream_bridge.py": "TurnSource.ELECTRON",
        "bridge/rex_quick_actions_bridge.py": "TurnSource.ELECTRON",
        "bridge/rex_voice_bridge.py": "TurnSource.VOICE",
        "rex/voice/loop.py": "TurnSource.VOICE",
        "rex/voice_loop_optimized.py": "TurnSource.VOICE",
        "rex/mobile_api/chat.py": "TurnSource.MOBILE",
        "rex/gui_app.py": "TurnSource.API",
        "rex/integrations/telegram/receiver.py": "TurnSource.TELEGRAM",
        "rex/telephony/twilio_handler.py": "TurnSource.TELEPHONY",
        "rex/mqtt_audio_router.py": "TurnSource.MQTT",
    }
    for relative, source_marker in expected.items():
        text = (root / relative).read_text(encoding="utf-8")
        assert "turn_invocation" in text, relative
        assert source_marker in text, relative
        assert "LanguageModel(" not in text, relative


def test_turn_invocation_rejects_blank_device_id() -> None:
    from rex.runtime.invocation import turn_invocation

    with pytest.raises(ValueError, match="device_id"):
        with turn_invocation(TurnSource.MOBILE, device_id=" "):
            pass


def _pipeline_assistant() -> Assistant:
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
    assistant._llm = MagicMock(model_name="test-model")
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
    router = MagicMock()
    router.route.return_value = IntentResult(handled=False, response=None, intent_type=None)
    assistant._intent_router = router
    assistant._turn_events = []
    assistant._turn_event_observer = assistant._turn_events.append

    async def dispatch(*_args, turn_events=None, **_kwargs):
        assert isinstance(turn_events, TurnEventStream)
        turn_events.emit(
            EventKind.CAPABILITY_PROGRESS,
            {"capability": "home_assistant", "status": "selected"},
        )
        turn_events.emit(
            EventKind.ACTION_PROGRESS,
            {"capability": "home_assistant", "status": "confirmation_required"},
        )
        return ActionResult(
            success=False,
            response="Please confirm locking the front door.",
            actions_taken=["home_assistant"],
        )

    assistant._action_dispatcher = MagicMock()
    assistant._action_dispatcher.dispatch = dispatch
    return assistant


def _semantic_trace(assistant: Assistant) -> list[tuple[EventKind, tuple[tuple[str, object], ...]]]:
    keep = {EventKind.ROUTE_PROGRESS, EventKind.CAPABILITY_PROGRESS, EventKind.ACTION_PROGRESS}
    return [
        (event.kind, tuple(sorted(event.details.items())))
        for event in assistant._turn_events
        if event.kind in keep
    ]


def test_same_authenticated_request_has_route_tool_verification_parity_across_interfaces() -> None:
    from rex.runtime.invocation import turn_invocation

    baseline = None
    for source, device_id in [
        (TurnSource.CLI, None),
        (TurnSource.ELECTRON, None),
        (TurnSource.VOICE, None),
        (TurnSource.MOBILE, "paired-phone"),
        (TurnSource.API, None),
    ]:
        assistant = _pipeline_assistant()
        with turn_invocation(source, device_id=device_id):
            reply = _run(assistant.generate_reply("lock the front door", active_user_id="james"))
        outcome = (reply, _semantic_trace(assistant))
        if baseline is None:
            baseline = outcome
        else:
            assert outcome == baseline
        assert assistant._turn_events[-1].kind is EventKind.COMPLETED


def test_canonical_voice_never_replaces_assistant_with_openclaw_brain() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "rex/voice/loop.py").read_text(encoding="utf-8")
    assert "VoiceBridge" not in text
    assert "get_openclaw_client" not in text
