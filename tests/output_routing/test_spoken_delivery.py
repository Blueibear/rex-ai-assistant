from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from rex.media.models import AudioTarget, MediaCapability, TargetKind
from rex.media.registry import AudioTargetRegistry
from rex.output_routing.models import UserOutputPolicy
from rex.output_routing.service import OutputRoutingService
from rex.output_routing.spoken import deliver_spoken_response

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
        capabilities=frozenset({MediaCapability.PLAY}),
        online=True,
        health="healthy",
    )


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
