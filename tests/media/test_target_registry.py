from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from rex.media.models import AudioTarget, MediaCapability, TargetKind
from rex.media.registry import AudioTargetRegistry


def _target(
    target_id: str,
    *,
    display_name: str,
    aliases: tuple[str, ...] = (),
    room: str | None = None,
    kind: TargetKind = TargetKind.SPEAKER,
    online: bool = True,
) -> AudioTarget:
    provider, native_id = target_id.split(":", 1)
    return AudioTarget(
        id=target_id,
        native_id=native_id,
        provider=provider,
        kind=kind,
        display_name=display_name,
        aliases=aliases,
        room=room,
        capabilities=frozenset({MediaCapability.PLAY, MediaCapability.PAUSE}),
        online=online,
        health="healthy" if online else "unavailable",
    )


def make_registry(
    *,
    two_living_room_targets: bool = False,
    origin_target_authorized: bool = True,
) -> AudioTargetRegistry:
    living_room = _target(
        "ha:media_player.living_room",
        display_name="Living Room Speaker",
        aliases=("downstairs", "sonos:RINCON_COLLISION"),
        room="Living Room",
    )
    kitchen = _target(
        "bose:kitchen",
        display_name="Kitchen Bose",
        aliases=("cooking speaker", "shared alias"),
        room="Kitchen",
    )
    collision = _target(
        "sonos:RINCON_COLLISION",
        display_name="Office Sonos",
        aliases=("desk speaker",),
        room="Office",
    )
    targets = [living_room, kitchen, collision]
    authorized_ids = {living_room.id, kitchen.id, collision.id}

    if two_living_room_targets:
        second_living_room = _target(
            "sonos:RINCON_2",
            display_name="Living Room Sonos",
            aliases=("shared alias",),
            room="living room",
        )
        targets.append(second_living_room)
        authorized_ids.add(second_living_room.id)

    if not origin_target_authorized:
        authorized_ids.remove(kitchen.id)

    return AudioTargetRegistry(
        targets,
        authorized_target_ids={"james": authorized_ids},
        origin_device_targets={"mic_kitchen": kitchen.id},
    )


def test_explicit_stable_id_wins_before_an_alias_collision() -> None:
    registry = make_registry()

    result = registry.resolve("sonos:RINCON_COLLISION", user_id="james")

    assert result.target is not None
    assert result.target.id == "sonos:RINCON_COLLISION"
    assert result.reason == "stable_id"


def test_exact_alias_is_normalized_without_fuzzy_matching() -> None:
    registry = make_registry()

    result = registry.resolve("  COOKING   SPEAKER ", user_id="james")

    assert result.target is not None
    assert result.target.id == "bose:kitchen"
    assert result.reason == "name_or_alias"


def test_unique_exact_room_resolves() -> None:
    registry = make_registry()

    result = registry.resolve("  living   ROOM ", user_id="james")

    assert result.target is not None
    assert result.target.id == "ha:media_player.living_room"
    assert result.reason == "room"


def test_ambiguous_room_does_not_guess() -> None:
    registry = make_registry(two_living_room_targets=True)

    result = registry.resolve("living room", user_id="james")

    assert result.target is None
    assert result.reason == "ambiguous"
    assert result.ambiguous_ids == (
        "ha:media_player.living_room",
        "sonos:RINCON_2",
    )


def test_ambiguous_exact_alias_does_not_guess() -> None:
    registry = make_registry(two_living_room_targets=True)

    result = registry.resolve("shared alias", user_id="james")

    assert result.target is None
    assert result.reason == "ambiguous"
    assert result.ambiguous_ids == ("bose:kitchen", "sonos:RINCON_2")


def test_unauthorized_target_is_filtered_before_unique_room_resolution() -> None:
    registry = make_registry(two_living_room_targets=True)
    registry = AudioTargetRegistry(
        registry.targets,
        authorized_target_ids={"james": {"sonos:RINCON_2"}},
    )

    result = registry.resolve("living room", user_id="james")

    assert result.target is not None
    assert result.target.id == "sonos:RINCON_2"


def test_explicit_unauthorized_target_is_not_returned() -> None:
    registry = make_registry(origin_target_authorized=False)

    result = registry.resolve("bose:kitchen", user_id="james")

    assert result.target is None
    assert result.reason == "not_authorized"


def test_offline_target_is_not_returned() -> None:
    offline = _target(
        "sonos:offline",
        display_name="Patio",
        room="Patio",
        online=False,
    )
    registry = AudioTargetRegistry(
        [offline],
        authorized_target_ids={"james": {offline.id}},
    )

    result = registry.resolve("sonos:offline", user_id="james")

    assert result.target is None
    assert result.reason == "offline"


def test_persistent_group_resolves_after_room_candidates() -> None:
    room_target = _target(
        "sonos:den",
        display_name="Den Speaker",
        room="Everywhere",
    )
    group = _target(
        "group:everywhere",
        display_name="Everywhere",
        kind=TargetKind.GROUP,
    )
    registry = AudioTargetRegistry(
        [group, room_target],
        authorized_target_ids={"james": {group.id, room_target.id}},
    )

    result = registry.resolve("everywhere", user_id="james")

    assert result.target == room_target
    assert result.reason == "room"


def test_persistent_group_resolves_by_exact_name() -> None:
    group = _target(
        "group:downstairs",
        display_name="Downstairs Group",
        aliases=("whole downstairs",),
        kind=TargetKind.GROUP,
    )
    registry = AudioTargetRegistry(
        [group],
        authorized_target_ids={"james": {group.id}},
    )

    result = registry.resolve("whole downstairs", user_id="james")

    assert result.target == group
    assert result.reason == "persistent_group"


def test_origin_device_is_used_when_query_is_absent() -> None:
    registry = make_registry()

    result = registry.resolve(None, user_id="james", origin_device_id="mic_kitchen")

    assert result.target is not None
    assert result.target.id == "bose:kitchen"
    assert result.reason == "origin_device"


def test_origin_is_only_a_preference_after_authorization() -> None:
    registry = make_registry(origin_target_authorized=False)

    result = registry.resolve(None, user_id="james", origin_device_id="mic_kitchen")

    assert result.target is None
    assert result.reason == "origin_not_authorized"


def test_partial_name_does_not_fuzzy_select() -> None:
    registry = make_registry()

    result = registry.resolve("living", user_id="james")

    assert result.target is None
    assert result.reason == "not_found"


def test_audio_target_and_registry_snapshot_are_immutable() -> None:
    registry = make_registry()
    target = registry.targets[0]

    with pytest.raises(FrozenInstanceError):
        target.display_name = "Changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        target.aliases[0] = "changed"  # type: ignore[index]

    assert isinstance(registry.targets, tuple)


def test_duplicate_stable_ids_are_rejected() -> None:
    target = _target("sonos:duplicate", display_name="One")
    duplicate = _target("sonos:duplicate", display_name="Two")

    with pytest.raises(ValueError, match="duplicate audio target id"):
        AudioTargetRegistry(
            [target, duplicate],
            authorized_target_ids={"james": {target.id}},
        )
