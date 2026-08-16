"""Task 5: exact conversational routing for canonical media tools."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from rex.actions.dispatcher import ActionDispatcher, ActionResult
from rex.context.builder import ContextPackage
from rex.intent.router import IntentResult
from rex.mobile_api.action_context import mobile_action_context, required_scope_for_tool
from rex.runtime.invocation import turn_invocation
from rex.runtime.turn import TurnSource


def _context(tool_context: str | None = None) -> ContextPackage:
    prompt = "user: test\nassistant:"
    if tool_context:
        prompt = f"{tool_context}\n{prompt}"
    return ContextPackage(
        messages=[{"role": "user", "content": "test"}],
        system_prompt="system",
        session_id="test",
        user_facts={},
        prompt=prompt,
    )


class _ContextBuilder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def build(self, transcript: str, **kwargs: object) -> ContextPackage:
        self.calls.append({"transcript": transcript, **kwargs})
        return _context(str(kwargs.get("tool_context") or ""))


class _LLM:
    def __init__(self, response: str = "canonical media response") -> None:
        self.response = response
        self.calls = 0

    def generate(self, *args: object, **kwargs: object) -> str:
        self.calls += 1
        return self.response


class _ResultHandler:
    async def process(self, transcript: str, completion: str, **kwargs: object) -> str:
        return completion


class _ExactToolDispatcher:
    def __init__(self) -> None:
        self.dispatch_calls: list[tuple[str, dict[str, object], dict[str, object]]] = []
        self.select_calls: list[str] = []

    def dispatch(
        self, name: str, args: dict[str, object], context: dict[str, object]
    ) -> SimpleNamespace:
        self.dispatch_calls.append((name, dict(args), dict(context)))
        return SimpleNamespace(success=True, output={"status": "verified", "target_id": "ha:x"})

    def format_tool_context(self, results: dict[str, object]) -> str:
        return f"MEDIA_RESULTS={results!r}"

    def select_tools_for_user(self, message: str, *, user_id: str) -> list[object]:
        self.select_calls.append(message)
        return []

    def execute_tools(self, *args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("exact media routing must not fan out to generic tools")

    def recovery_plan(self, *args: object, **kwargs: object) -> None:
        return None


def _dispatcher(
    tools: _ExactToolDispatcher,
    *,
    music_handler: object | None = None,
    ha_bridge: object | None = None,
) -> ActionDispatcher:
    return ActionDispatcher(
        context_builder=_ContextBuilder(),
        llm=_LLM(),
        result_handler=_ResultHandler(),
        tool_dispatcher=tools,
        music_handler=music_handler,
        ha_bridge=ha_bridge,
    )


def _intent() -> IntentResult:
    return IntentResult(handled=False, response=None, intent_type=None)


def _run(dispatcher: ActionDispatcher, transcript: str) -> ActionResult:
    return asyncio.run(dispatcher.dispatch(_intent(), _context(), transcript, user_id="james"))


def test_play_uses_exact_media_manage_with_trusted_origin() -> None:
    tools = _ExactToolDispatcher()
    legacy = MagicMock()
    legacy.handle.return_value = "legacy direct mutation"
    ha = MagicMock()
    ha.enabled = True
    ha.process_transcript.return_value = "HA fanout"
    dispatcher = _dispatcher(tools, music_handler=legacy, ha_bridge=ha)

    with turn_invocation(TurnSource.VOICE, device_id="mic_kitchen"):
        result = _run(dispatcher, "play jazz")

    assert result.response == "canonical media response"
    assert tools.dispatch_calls == [
        (
            "media_manage",
            {"transcript": "play jazz", "origin_device_id": "mic_kitchen"},
            {"user_id": "james"},
        )
    ]
    legacy.handle.assert_not_called()
    assert tools.select_calls == []
    ha.process_transcript.assert_not_called()


def test_media_read_routes_exactly_without_generic_or_ha_fanout() -> None:
    tools = _ExactToolDispatcher()
    ha = MagicMock()
    ha.enabled = True
    ha.process_transcript.return_value = "HA fanout"
    dispatcher = _dispatcher(tools, ha_bridge=ha)

    with turn_invocation(TurnSource.VOICE, device_id="mic_kitchen"):
        _run(dispatcher, "what's playing")

    assert tools.dispatch_calls == [
        (
            "media_read",
            {"transcript": "what's playing", "origin_device_id": "mic_kitchen"},
            {"user_id": "james"},
        )
    ]
    assert tools.select_calls == []
    ha.process_transcript.assert_not_called()


def test_followup_transfer_is_exact_manage_and_never_fans_out() -> None:
    tools = _ExactToolDispatcher()
    ha = MagicMock()
    ha.enabled = True
    dispatcher = _dispatcher(tools, ha_bridge=ha)

    _run(dispatcher, "move it to the living room")

    assert [call[0] for call in tools.dispatch_calls] == ["media_manage"]
    assert tools.select_calls == []
    ha.process_transcript.assert_not_called()


def test_mobile_media_mutation_is_recognized_but_not_pre_dispatched() -> None:
    tools = _ExactToolDispatcher()
    legacy = MagicMock()
    legacy.handle.return_value = "legacy direct mutation"
    dispatcher = _dispatcher(tools, music_handler=legacy)

    with mobile_action_context({"home.control"}, permissions={"ha_control"}):
        _run(dispatcher, "pause")

    assert tools.dispatch_calls == []
    assert tools.select_calls == []
    legacy.handle.assert_not_called()


def test_mobile_scope_mapping_covers_canonical_media_tools() -> None:
    assert required_scope_for_tool("media_read", operation="read") == "home.read"
    assert required_scope_for_tool("media_manage", operation="mutation") == "home.control"
    assert required_scope_for_tool("music_play", operation="mutation") is None


class _FakeHABridge:
    def list_entities(self) -> list[dict[str, object]]:
        return [
            {
                "entity_id": "media_player.kitchen",
                "state": "playing",
                "attributes": {"friendly_name": "Kitchen"},
            }
        ]

    def get_entity_state(self, entity_id: str) -> dict[str, object] | None:
        assert entity_id == "media_player.kitchen"
        return {
            "entity_id": entity_id,
            "state": "playing",
            "attributes": {"friendly_name": "Kitchen", "media_title": "Jazz"},
        }

    def execute_media_service(
        self, entity_id: str, service: str, *, volume_level: float | None = None
    ) -> tuple[bool, str]:
        return True, "accepted"


def test_assistant_configures_authorized_media_service_with_exact_origin(
    monkeypatch,
) -> None:
    from rex.assistant import Assistant
    from rex.media.parser import MediaCommand

    assistant: Any = Assistant.__new__(Assistant)
    assistant._settings = SimpleNamespace(device_room_map={"mic_kitchen": "kitchen"})
    assistant._user_id = "james"
    assistant._ha_bridge = _FakeHABridge()

    monkeypatch.setattr("rex.identity.list_known_users", lambda: [{"id": "james"}, {"id": "cole"}])
    monkeypatch.setattr(
        "rex.permissions.get_permissions",
        lambda user_id: ["ha_control"] if user_id == "james" else [],
    )
    monkeypatch.setattr("rex.media.tools.set_media_service", lambda service: None)

    assistant._configure_media_service()

    result = assistant._media_service.execute(
        MediaCommand(action="state"),
        user_id="james",
        origin_device_id="mic_kitchen",
    )
    assert result.outcome == "read"
    assert result.requested_target_id == "ha:media_player.kitchen"

    denied = assistant._media_service.execute(
        MediaCommand(action="state", target_text="ha:media_player.kitchen"),
        user_id="cole",
    )
    assert denied.outcome == "not_authorized"


def test_exact_media_turn_blocks_post_llm_inline_ha_bypass() -> None:
    from rex.tools.result_handler import ToolResultHandler

    tools = _ExactToolDispatcher()
    ha = MagicMock()
    ha.enabled = True
    ha.post_process_response.return_value = "HA BYPASS EXECUTED"
    result_handler = ToolResultHandler(
        tool_router_fn=lambda completion, *_args: completion,
        ha_bridge=ha,
    )
    dispatcher = ActionDispatcher(
        context_builder=_ContextBuilder(),
        llm=_LLM("[[ha:media_player.media_pause entity_id=media_player.kitchen]]"),
        result_handler=result_handler,
        tool_dispatcher=tools,
        ha_bridge=ha,
    )

    result = _run(dispatcher, "pause in the kitchen")

    assert "[[ha:" not in result.response.casefold()
    assert result.response != "HA BYPASS EXECUTED"
    ha.post_process_response.assert_not_called()


class _ChangingHABridge(_FakeHABridge):
    def __init__(self) -> None:
        self.entity = "media_player.kitchen"

    def list_entities(self) -> list[dict[str, object]]:
        name = self.entity.rsplit(".", 1)[-1].replace("_", " ").title()
        return [
            {"entity_id": self.entity, "state": "playing", "attributes": {"friendly_name": name}}
        ]

    def get_entity_state(self, entity_id: str) -> dict[str, object] | None:
        if entity_id != self.entity:
            return None
        return {"entity_id": entity_id, "state": "playing", "attributes": {"media_title": "Jazz"}}


def test_assistant_media_service_refreshes_dynamic_targets(monkeypatch) -> None:
    from rex.assistant import Assistant
    from rex.media.parser import MediaCommand

    bridge = _ChangingHABridge()
    assistant: Any = Assistant.__new__(Assistant)
    assistant._settings = SimpleNamespace(device_room_map={})
    assistant._user_id = "james"
    assistant._ha_bridge = bridge
    monkeypatch.setattr("rex.identity.list_known_users", lambda: [{"id": "james"}])
    monkeypatch.setattr("rex.permissions.get_permissions", lambda _user_id: ["ha_control"])
    monkeypatch.setattr("rex.media.tools.set_media_service", lambda service: None)

    assistant._configure_media_service()
    bridge.entity = "media_player.den"
    result = assistant._media_service.execute(
        MediaCommand(action="state", target_text="ha:media_player.den"), user_id="james"
    )

    assert result.outcome == "read"
    assert result.requested_target_id == "ha:media_player.den"


def test_assistant_media_registry_includes_persistent_groups(monkeypatch, tmp_path) -> None:
    import json

    from rex.assistant import Assistant
    from rex.media.parser import MediaCommand

    group_path = tmp_path / "media" / "speaker_groups.json"
    group_path.parent.mkdir(parents=True)
    group_path.write_text(
        json.dumps(
            {
                "version": 1,
                "groups": [
                    {
                        "id": "group:downstairs",
                        "name": "Downstairs",
                        "member_ids": ["ha:media_player.kitchen"],
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(
        "rex.media.groups.household_data_path", lambda *parts: tmp_path.joinpath(*parts)
    )
    monkeypatch.setattr("rex.identity.list_known_users", lambda: [{"id": "james"}])
    monkeypatch.setattr("rex.permissions.get_permissions", lambda _user_id: ["ha_control"])
    monkeypatch.setattr("rex.media.tools.set_media_service", lambda service: None)

    assistant: Any = Assistant.__new__(Assistant)
    assistant._settings = SimpleNamespace(device_room_map={})
    assistant._user_id = "james"
    assistant._ha_bridge = _FakeHABridge()
    assistant._configure_media_service()
    result = assistant._media_service.execute(
        MediaCommand(action="state", target_text="group:downstairs"), user_id="james"
    )
    assert result.outcome == "unsupported"
    assert result.requested_target_id == "group:downstairs"


def test_assistant_media_registry_includes_cached_local_speakers(monkeypatch) -> None:
    from rex.assistant import Assistant
    from rex.audio.speaker_discovery import DiscoveredSpeaker
    from rex.media.parser import MediaCommand

    discovery = MagicMock()
    discovery.get_cached_speakers.return_value = [
        DiscoveredSpeaker(provider="sonos", name="Office", ip="10.0.0.8", model="One")
    ]
    monkeypatch.setattr("rex.audio.speaker_discovery.get_speaker_discovery", lambda: discovery)
    monkeypatch.setattr("rex.identity.list_known_users", lambda: [{"id": "james"}])
    monkeypatch.setattr("rex.permissions.get_permissions", lambda _user_id: ["ha_control"])
    monkeypatch.setattr("rex.media.tools.set_media_service", lambda service: None)

    assistant: Any = Assistant.__new__(Assistant)
    assistant._settings = SimpleNamespace(device_room_map={})
    assistant._user_id = "james"
    assistant._ha_bridge = None
    assistant._configure_media_service()
    result = assistant._media_service.execute(
        MediaCommand(action="state", target_text="sonos:10.0.0.8"), user_id="james"
    )
    assert result.outcome == "unsupported"
    assert result.requested_target_id == "sonos:10.0.0.8"
