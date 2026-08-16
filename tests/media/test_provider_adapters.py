from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from rex.audio.speaker_discovery import DiscoveredSpeaker
from rex.ha_bridge import HABridge
from rex.integrations.music_assistant import MusicAssistantClient
from rex.media.adapters import (
    HomeAssistantMediaAdapter,
    MusicAssistantAdapter,
    SmartSpeakerAdapter,
)
from rex.media.models import (
    AudioTarget,
    MediaAction,
    MediaCapability,
    MediaState,
    TargetKind,
)


class _FakeSpeakerDiscovery:
    def __init__(self, speakers: list[DiscoveredSpeaker]) -> None:
        self._speakers = speakers
        self.discovery_calls = 0

    def discover_now(self) -> list[DiscoveredSpeaker]:
        self.discovery_calls += 1
        return list(self._speakers)


class _FakeHABridge:
    def __init__(self, states: list[dict[str, Any]]) -> None:
        self.states = states
        self.intents: list[Any] = []

    def list_entities(self) -> list[dict[str, Any]]:
        return deepcopy(self.states)

    def get_entity_state(self, entity_id: str) -> dict[str, Any] | None:
        for state in self.states:
            if state["entity_id"] == entity_id:
                return deepcopy(state)
        return None

    def execute_media_service(
        self,
        entity_id: str,
        service: str,
        *,
        volume_level: float | None = None,
    ) -> tuple[bool, str]:
        self.intents.append((entity_id, service, volume_level))
        return True, "accepted"


class _FakeMusicAssistantClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, str | None]] = []

    def play(self, query: str, room: str | None = None) -> dict[str, bool]:
        self.calls.append(("play", query, room))
        return {"ok": True}

    def pause(self, room: str | None = None) -> dict[str, bool]:
        self.calls.append(("pause", None, room))
        return {"ok": True}

    def resume(self, room: str | None = None) -> dict[str, bool]:
        self.calls.append(("resume", None, room))
        return {"ok": True}

    def skip(self, room: str | None = None) -> dict[str, bool]:
        self.calls.append(("skip", None, room))
        return {"ok": True}

    def set_volume(self, level: int, room: str | None = None) -> dict[str, bool]:
        self.calls.append(("set_volume", level, room))
        return {"ok": True}


def _ha_state(
    entity_id: str,
    *,
    state: str = "idle",
    friendly_name: str | None = None,
    **attributes: Any,
) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "state": state,
        "attributes": {
            "friendly_name": friendly_name or entity_id,
            **attributes,
        },
    }


def _music_assistant_target() -> AudioTarget:
    return AudioTarget(
        id="music_assistant:kitchen",
        native_id="kitchen",
        provider="music_assistant",
        kind=TargetKind.SPEAKER,
        display_name="Kitchen",
        aliases=(),
        room=None,
        capabilities=frozenset(
            {
                MediaCapability.PLAY,
                MediaCapability.PAUSE,
                MediaCapability.RESUME,
                MediaCapability.NEXT,
                MediaCapability.SET_VOLUME,
            }
        ),
        online=True,
        health="healthy",
    )


def test_ha_adapter_discovers_only_media_players_without_user_authorization() -> None:
    adapter = HomeAssistantMediaAdapter(
        _FakeHABridge(
            [
                _ha_state("light.desk", state="on", friendly_name="Desk"),
                _ha_state(
                    "media_player.den",
                    state="playing",
                    friendly_name="Den Speaker",
                ),
            ]
        )
    )

    targets = adapter.discover_targets()

    assert [target.native_id for target in targets] == ["media_player.den"]
    assert targets[0].id == "ha:media_player.den"
    assert targets[0].provider == "ha"
    assert targets[0].display_name == "Den Speaker"
    assert targets[0].capabilities == frozenset(
        {
            MediaCapability.PLAY,
            MediaCapability.PAUSE,
            MediaCapability.NEXT,
            MediaCapability.SET_VOLUME,
        }
    )


def test_ha_bridge_preserves_entity_state_and_reuses_existing_state_reader(
    monkeypatch,
) -> None:
    bridge = HABridge.__new__(HABridge)
    bridge._entity_states = {
        "media_player.den": _ha_state(
            "media_player.den",
            state="playing",
            friendly_name="Den Speaker",
        )
    }
    bridge._base_url = "http://ha.local:8123"
    bridge._token = "token"
    bridge._verify_ssl = True
    bridge._timeout = 5.0
    monkeypatch.setattr(bridge, "_refresh_entity_cache", lambda force=False: None)
    state_reads: list[str] = []

    def fake_get_device_state(
        entity_id: str,
        base_url: str,
        token: str,
        *,
        verify_ssl: bool,
        timeout: float,
    ) -> dict[str, Any]:
        state_reads.append(entity_id)
        assert base_url == "http://ha.local:8123"
        assert token == "token"
        assert verify_ssl is True
        assert timeout == 5.0
        return _ha_state(entity_id, state="paused")

    monkeypatch.setattr("rex.ha_bridge.get_device_state", fake_get_device_state)

    entities = bridge.list_entities()
    state = bridge.get_entity_state("media_player.den")

    assert entities[0]["friendly_name"] == "Den Speaker"
    assert entities[0]["state"] == "playing"
    assert entities[0]["attributes"]["friendly_name"] == "Den Speaker"
    assert state is not None
    assert state["state"] == "paused"
    assert state_reads == ["media_player.den"]


def test_ha_bridge_lists_entity_without_friendly_name(monkeypatch) -> None:
    bridge = HABridge(base_url="http://ha.local:8123", token="token")
    response = {
        "entity_id": "media_player.den",
        "state": "unknown",
        "attributes": {"volume_level": 0.25},
    }
    monkeypatch.setattr(bridge, "_request", lambda method, path: [deepcopy(response)])

    entities = bridge.list_entities()

    assert entities == [
        {
            **response,
            "friendly_name": "media_player.den",
            "attributes": {"volume_level": 0.25},
        }
    ]


def test_ha_bridge_public_media_wrapper_is_narrow(monkeypatch) -> None:
    bridge = HABridge.__new__(HABridge)
    intents: list[Any] = []

    def execute_intent(intent: Any) -> tuple[bool, str]:
        intents.append(intent)
        return True, "accepted"

    monkeypatch.setattr(bridge, "_execute_intent", execute_intent)

    result = bridge.execute_media_service(
        "media_player.den",
        "volume_set",
        volume_level=0.35,
    )

    assert result == (True, "accepted")
    assert len(intents) == 1
    assert intents[0].domain == "media_player"
    assert intents[0].service == "volume_set"
    assert intents[0].entity_id == "media_player.den"
    assert intents[0].data == {
        "entity_id": "media_player.den",
        "volume_level": 0.35,
    }

    with pytest.raises(ValueError, match="Unsupported Home Assistant media service"):
        bridge.execute_media_service("media_player.den", "media_stop")

    assert len(intents) == 1


def test_ha_adapter_uses_existing_mutation_and_independent_state_paths() -> None:
    bridge = _FakeHABridge(
        [
            _ha_state(
                "media_player.den",
                state="paused",
                friendly_name="Den Speaker",
                volume_level=0.42,
                media_position=18.5,
                media_content_id="track:7",
                media_title="Seven",
            )
        ]
    )
    adapter = HomeAssistantMediaAdapter(bridge)
    target = adapter.discover_targets()[0]

    acknowledgement = adapter.execute_action(target, MediaAction.SET_VOLUME, value=35)
    snapshot = adapter.get_state(target)

    assert acknowledgement.accepted is True
    assert len(bridge.intents) == 1
    assert bridge.intents[0] == ("media_player.den", "volume_set", 0.35)
    assert snapshot.target_id == "ha:media_player.den"
    assert snapshot.playback is MediaState.PAUSED
    assert snapshot.volume_percent == 42.0
    assert snapshot.position_seconds == 18.5
    assert snapshot.current_item_id == "track:7"
    assert snapshot.current_item_title == "Seven"


def test_ha_adapter_rejects_operations_outside_existing_rex_media_paths() -> None:
    bridge = _FakeHABridge([_ha_state("media_player.den")])
    adapter = HomeAssistantMediaAdapter(bridge)
    target = adapter.discover_targets()[0]

    acknowledgement = adapter.execute_action(target, MediaAction.PREVIOUS)

    assert acknowledgement.accepted is False
    assert acknowledgement.detail == "Home Assistant action previous is unsupported"
    assert bridge.intents == []


def test_ha_adapter_observed_unknown_entity_remains_online() -> None:
    adapter = HomeAssistantMediaAdapter(
        _FakeHABridge([_ha_state("media_player.den", state="unknown")])
    )

    target = adapter.discover_targets()[0]

    assert target.online is True
    assert target.health == "unknown"


def test_ha_adapter_failed_state_read_is_unknown_not_unavailable() -> None:
    bridge = _FakeHABridge([_ha_state("media_player.den")])
    adapter = HomeAssistantMediaAdapter(bridge)
    target = adapter.discover_targets()[0]
    bridge.states.clear()

    snapshot = adapter.get_state(target)

    assert snapshot.playback is MediaState.UNKNOWN


def test_smart_speaker_adapter_is_discovery_only() -> None:
    discovery = _FakeSpeakerDiscovery(
        [
            DiscoveredSpeaker(
                provider="sonos",
                name="Office Sonos",
                ip="192.168.1.40",
                model="Era 100",
            ),
            DiscoveredSpeaker(
                provider="bose",
                name="Kitchen Bose",
                ip="192.168.1.20",
                model="SoundTouch 10",
            ),
        ]
    )
    adapter = SmartSpeakerAdapter(discovery)

    targets = adapter.discover_targets()
    acknowledgement = adapter.execute_action(targets[0], MediaAction.PLAY, value="track:7")
    snapshot = adapter.get_state(targets[0])

    assert [target.id for target in targets] == [
        "bose:192.168.1.20",
        "sonos:192.168.1.40",
    ]
    assert all(target.capabilities == frozenset() for target in targets)
    assert all(target.online is True for target in targets)
    assert discovery.discovery_calls == 1
    assert acknowledgement.accepted is False
    assert acknowledgement.detail == "Smart-speaker media mutations are unsupported"
    assert snapshot.playback is MediaState.UNKNOWN


def test_discovered_speaker_does_not_expose_canonical_target_id() -> None:
    speaker = DiscoveredSpeaker(
        provider="sonos",
        name="Office Sonos",
        ip="192.168.1.40",
        model="Era 100",
    )

    assert not hasattr(speaker, "target_id")


def test_music_assistant_adapter_does_not_invent_discovery_or_state() -> None:
    adapter = MusicAssistantAdapter(_FakeMusicAssistantClient())
    target = _music_assistant_target()

    targets = adapter.discover_targets()
    snapshot = adapter.get_state(target)

    assert targets == ()
    assert snapshot.target_id == "music_assistant:kitchen"
    assert snapshot.playback is MediaState.UNKNOWN
    assert snapshot.volume_percent is None


def test_music_assistant_adapter_wraps_only_existing_client_mutations() -> None:
    client = _FakeMusicAssistantClient()
    adapter = MusicAssistantAdapter(client)
    target = _music_assistant_target()

    play_ack = adapter.execute_action(target, MediaAction.PLAY, value="Kind of Blue")
    pause_ack = adapter.execute_action(target, MediaAction.PAUSE)
    resume_ack = adapter.execute_action(target, MediaAction.RESUME)
    next_ack = adapter.execute_action(target, MediaAction.NEXT)
    volume_ack = adapter.execute_action(target, MediaAction.SET_VOLUME, value=55)
    unsupported_ack = adapter.execute_action(target, MediaAction.PREVIOUS)

    assert play_ack.accepted is True
    assert pause_ack.accepted is True
    assert resume_ack.accepted is True
    assert next_ack.accepted is True
    assert volume_ack.accepted is True
    assert unsupported_ack.accepted is False
    assert unsupported_ack.detail == "Music Assistant action previous is unsupported"
    assert client.calls == [
        ("play", "Kind of Blue", "kitchen"),
        ("pause", None, "kitchen"),
        ("resume", None, "kitchen"),
        ("skip", None, "kitchen"),
        ("set_volume", 55, "kitchen"),
    ]


def test_music_assistant_play_requires_a_query() -> None:
    client = _FakeMusicAssistantClient()
    adapter = MusicAssistantAdapter(client)

    acknowledgement = adapter.execute_action(_music_assistant_target(), MediaAction.PLAY)

    assert acknowledgement.accepted is False
    assert acknowledgement.detail == "Music Assistant play requires a query"
    assert client.calls == []


def test_music_assistant_client_does_not_expose_adapter_action_policy() -> None:
    client = MusicAssistantClient()

    assert not hasattr(client, "supported_adapter_actions")
