from __future__ import annotations

from datetime import UTC, datetime

from rex.media.models import AudioTarget, MediaCapability, TargetKind
from rex.media.parser import MediaCommand, MediaCommandAction
from rex.media.registry import AudioTargetRegistry
from rex.output_routing.execution import resolve_media_command, resolve_spoken_response
from rex.output_routing.models import OutputKind, UserOutputPolicy
from rex.output_routing.service import OutputRoutingService

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


def test_spoken_response_prefers_authorized_origin(tmp_path) -> None:
    den = _target("local:den", "Den")
    registry = AudioTargetRegistry(
        (den,),
        authorized_target_ids={"james": {den.id}},
        origin_device_targets={"mic_den": den.id},
    )
    routing = OutputRoutingService(registry, root=tmp_path)

    route = resolve_spoken_response(
        routing,
        user_id="james",
        origin_device_id="mic_den",
        at=NOW,
    )

    assert route.output_kind is OutputKind.SPOKEN_RESPONSE
    assert route.target_id == den.id
    assert route.reason == "request_origin"


def test_media_command_without_explicit_target_uses_user_policy(tmp_path) -> None:
    kitchen = _target("ha:media_player.kitchen", "Kitchen")
    registry = AudioTargetRegistry(
        (kitchen,),
        authorized_target_ids={"james": {kitchen.id}},
    )
    routing = OutputRoutingService(registry, root=tmp_path)
    routing.save_policy("james", UserOutputPolicy(media_target_id=kitchen.id))
    command = MediaCommand(action=MediaCommandAction.PLAY, query="Miles Davis")

    routed = resolve_media_command(
        routing,
        command,
        user_id="james",
        origin_device_id=None,
        at=NOW,
    )

    assert routed.target_text == kitchen.id
    assert routed.query == "Miles Davis"


def test_explicit_media_target_is_not_replaced_by_stored_default(tmp_path) -> None:
    kitchen = _target("ha:media_player.kitchen", "Kitchen")
    den = _target("ha:media_player.den", "Den")
    registry = AudioTargetRegistry(
        (kitchen, den),
        authorized_target_ids={"james": {kitchen.id, den.id}},
    )
    routing = OutputRoutingService(registry, root=tmp_path)
    routing.save_policy("james", UserOutputPolicy(media_target_id=kitchen.id))
    command = MediaCommand(
        action=MediaCommandAction.PLAY,
        query="Miles Davis",
        target_text="Den",
    )

    routed = resolve_media_command(
        routing,
        command,
        user_id="james",
        origin_device_id=None,
        at=NOW,
    )

    assert routed.target_text == "Den"
