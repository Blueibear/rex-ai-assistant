from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from rex.media.models import AudioTarget, MediaCapability, TargetKind
from rex.media.registry import AudioTargetRegistry
from rex.output_routing.models import UserOutputPolicy
from rex.output_routing.service import OutputRoutingService
from rex.output_routing.spoken import (
    HomeAssistantSpokenVolumeController,
    deliver_spoken_response,
)

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


def _target(target_id: str, name: str) -> AudioTarget:
    return AudioTarget(
        id=target_id,
        native_id=target_id.split(":", 1)[-1],
        provider=target_id.split(":", 1)[0],
        kind=TargetKind.SPEAKER,
        display_name=name,
        aliases=(),
        room=None,
        capabilities=frozenset({MediaCapability.PLAY, MediaCapability.SET_VOLUME}),
        online=True,
        health="healthy",
    )


class _RecordingVolumeController:
    def __init__(self, events: list[tuple[str, object]], *, volume: float = 20.0) -> None:
        self.events = events
        self.volume = volume
        self.fail_restore = False
        self._sets = 0

    def read_volume(self, target_id: str) -> float | None:
        self.events.append(("read", target_id))
        return self.volume

    def set_and_verify_volume(self, target_id: str, volume: int | float) -> bool:
        self._sets += 1
        self.events.append(("volume", float(volume)))
        if self.fail_restore and self._sets > 1:
            return False
        self.volume = float(volume)
        return True


class _FakeHABridge:
    def __init__(self) -> None:
        self.volume = 0.25
        self.calls: list[tuple[str, str, float | None]] = []

    def get_entity_state(self, entity_id: str) -> dict[str, object]:
        return {"entity_id": entity_id, "attributes": {"volume_level": self.volume}}

    def execute_media_service(
        self,
        entity_id: str,
        service: str,
        *,
        volume_level: float | None = None,
        is_volume_muted: bool | None = None,
    ) -> tuple[bool, str]:
        del is_volume_muted
        self.calls.append((entity_id, service, volume_level))
        assert volume_level is not None
        self.volume = volume_level
        return True, "ok"


def test_spoken_delivery_uses_remote_target_without_local_duplicate(tmp_path) -> None:
    kitchen = _target("ha:media_player.kitchen", "Kitchen")
    registry = AudioTargetRegistry(
        (kitchen,), authorized_target_ids={"james": {kitchen.id}}
    )
    routing = OutputRoutingService(registry, root=tmp_path)
    routing.save_policy("james", UserOutputPolicy(spoken_response_target_id=kitchen.id))
    remote: list[tuple[str, str]] = []
    local: list[str] = []

    result = asyncio.run(
        deliver_spoken_response(
            "Dinner is ready.",
            routing=routing,
            user_id="james",
            origin_device_id=None,
            at=NOW,
            remote_sender=lambda target_id, text: remote.append((target_id, text)) or True,
            local_speak=lambda text: local.append(text),
        )
    )

    assert result.target_id == kitchen.id
    assert result.delivered is True
    assert remote == [(kitchen.id, "Dinner is ready.")]
    assert local == []


def test_spoken_delivery_applies_and_restores_verified_route_volume(tmp_path) -> None:
    kitchen = _target("ha:media_player.kitchen", "Kitchen")
    registry = AudioTargetRegistry(
        (kitchen,), authorized_target_ids={"james": {kitchen.id}}
    )
    routing = OutputRoutingService(registry, root=tmp_path)
    routing.save_policy(
        "james",
        UserOutputPolicy(
            spoken_response_target_id=kitchen.id,
            spoken_response_volume=55,
        ),
    )
    events: list[tuple[str, object]] = []
    controller = _RecordingVolumeController(events)

    result = asyncio.run(
        deliver_spoken_response(
            "Dinner is ready.",
            routing=routing,
            user_id="james",
            origin_device_id=None,
            at=NOW,
            remote_sender=lambda target_id, text: events.append(("speak", text)) or True,
            local_speak=lambda _text: None,
            volume_controller=controller,
        )
    )

    assert result.delivered is True
    assert result.target_volume == 55
    assert events == [
        ("read", kitchen.id),
        ("volume", 55.0),
        ("speak", "Dinner is ready."),
        ("volume", 20.0),
    ]
    assert controller.volume == 20.0


def test_spoken_delivery_reports_unverified_volume_restoration(tmp_path) -> None:
    kitchen = _target("ha:media_player.kitchen", "Kitchen")
    registry = AudioTargetRegistry(
        (kitchen,), authorized_target_ids={"james": {kitchen.id}}
    )
    routing = OutputRoutingService(registry, root=tmp_path)
    routing.save_policy(
        "james",
        UserOutputPolicy(
            spoken_response_target_id=kitchen.id,
            spoken_response_volume=55,
        ),
    )
    events: list[tuple[str, object]] = []
    controller = _RecordingVolumeController(events)
    controller.fail_restore = True

    result = asyncio.run(
        deliver_spoken_response(
            "Dinner is ready.",
            routing=routing,
            user_id="james",
            origin_device_id=None,
            at=NOW,
            remote_sender=lambda _target_id, _text: True,
            local_speak=lambda _text: None,
            volume_controller=controller,
        )
    )

    assert result.delivered is True
    assert result.reason == "delivered_volume_restore_unverified"


def test_home_assistant_volume_controller_sets_and_rereads_volume() -> None:
    bridge = _FakeHABridge()
    controller = HomeAssistantSpokenVolumeController(lambda: bridge)
    target_id = "ha:media_player.kitchen"

    assert controller.read_volume(target_id) == 25.0
    assert controller.set_and_verify_volume(target_id, 60) is True
    assert controller.read_volume(target_id) == 60.0
    assert bridge.calls == [("media_player.kitchen", "volume_set", 0.6)]


def test_spoken_delivery_preserves_local_voice_when_no_route_is_configured(tmp_path) -> None:
    registry = AudioTargetRegistry((), authorized_target_ids={"james": set()})
    routing = OutputRoutingService(registry, root=tmp_path)
    local: list[str] = []

    result = asyncio.run(
        deliver_spoken_response(
            "Hello.",
            routing=routing,
            user_id="james",
            origin_device_id=None,
            at=NOW,
            remote_sender=lambda _target_id, _text: False,
            local_speak=lambda text: local.append(text),
        )
    )

    assert result.delivered is True
    assert result.reason == "local_default"
    assert local == ["Hello."]
