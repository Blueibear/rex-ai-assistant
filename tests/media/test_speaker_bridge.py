from __future__ import annotations

import json
from pathlib import Path

from bridge.rex_speaker_bridge import handle_speaker_request
from rex.media.groups import SpeakerGroupStore
from rex.media.models import AudioTarget, MediaCapability, TargetKind
from rex.media.registry import AudioTargetRegistry


def _target(
    target_id: str,
    name: str,
    *,
    room: str | None = None,
    online: bool = True,
    capabilities: frozenset[MediaCapability] | None = None,
) -> AudioTarget:
    provider, native = target_id.split(":", 1)
    return AudioTarget(
        id=target_id,
        native_id=native,
        provider=provider,
        kind=TargetKind.SPEAKER,
        display_name=name,
        aliases=(),
        room=room,
        capabilities=capabilities or frozenset({MediaCapability.PLAY, MediaCapability.PAUSE}),
        online=online,
        health="ok" if online else "offline",
    )


def _registry(*targets: AudioTarget) -> AudioTargetRegistry:
    return AudioTargetRegistry(
        targets,
        authorized_target_ids={
            "james": {target.id for target in targets},
            "cole": {target.id for target in targets if target.id.endswith("bedroom")},
        },
    )


def _groups(tmp_path: Path, registry: AudioTargetRegistry) -> SpeakerGroupStore:
    by_id = {target.id: target for target in registry.targets}
    return SpeakerGroupStore(
        tmp_path / "groups.json",
        target_exists=by_id.__contains__,
        target_capabilities=lambda target_id: by_id[target_id].capabilities,
    )


def test_list_targets_filters_to_bound_user_and_never_exposes_credentials(tmp_path: Path) -> None:
    kitchen = _target("ha:media_player.kitchen", "Kitchen", room="Kitchen")
    bedroom = _target("ha:media_player.bedroom", "Bedroom", room="Bedroom", online=False)
    registry = _registry(kitchen, bedroom)

    body, code = handle_speaker_request(
        {"command": "list_targets", "user": "james", "data_scope": "private"},
        registry=registry,
        group_store=_groups(tmp_path, registry),
        bound_user_id="cole",
    )

    assert code == 0
    assert body["ok"] is True
    assert [target["id"] for target in body["targets"]] == ["ha:media_player.bedroom"]
    serialized = json.dumps(body)
    assert "credential_ref" not in serialized
    assert "token" not in serialized.casefold()
    assert body["targets"][0]["online"] is False


def test_refresh_targets_uses_fresh_registry_snapshot(tmp_path: Path) -> None:
    old = _target("ha:media_player.kitchen", "Kitchen")
    new = _target("ha:media_player.den", "Den")
    registry = _registry(old)
    refreshed = _registry(new)
    calls = 0

    def refresh() -> AudioTargetRegistry:
        nonlocal calls
        calls += 1
        return refreshed

    body, code = handle_speaker_request(
        {"command": "refresh_targets", "data_scope": "private"},
        registry=registry,
        group_store=_groups(tmp_path, registry),
        bound_user_id="james",
        refresh_registry=refresh,
    )

    assert code == 0
    assert calls == 1
    assert [target["id"] for target in body["targets"]] == ["ha:media_player.den"]


def test_group_crud_is_bound_to_authorized_visible_members(tmp_path: Path) -> None:
    kitchen = _target("ha:media_player.kitchen", "Kitchen")
    bedroom = _target("ha:media_player.bedroom", "Bedroom")
    registry = _registry(kitchen, bedroom)
    groups = _groups(tmp_path, registry)

    created, code = handle_speaker_request(
        {
            "command": "create_group",
            "name": "Downstairs",
            "member_ids": [kitchen.id, bedroom.id],
            "data_scope": "private",
        },
        registry=registry,
        group_store=groups,
        bound_user_id="james",
    )
    assert code == 0
    group_id = created["group"]["id"]
    assert created["group"]["capabilities"] == []
    assert created["lifecycle"]["state"] == "verified"

    renamed, code = handle_speaker_request(
        {"command": "rename_group", "group_id": group_id, "name": "Main Floor"},
        registry=registry,
        group_store=groups,
        bound_user_id="james",
    )
    assert code == 0
    assert renamed["group"]["name"] == "Main Floor"
    assert renamed["lifecycle"]["state"] == "verified"

    changed, code = handle_speaker_request(
        {"command": "set_group_members", "group_id": group_id, "member_ids": [kitchen.id]},
        registry=registry,
        group_store=groups,
        bound_user_id="james",
    )
    assert code == 0
    assert changed["group"]["member_ids"] == [kitchen.id]
    assert changed["lifecycle"]["state"] == "verified"

    listed, code = handle_speaker_request(
        {"command": "list_groups"},
        registry=registry,
        group_store=groups,
        bound_user_id="james",
    )
    assert code == 0
    assert [group["id"] for group in listed["groups"]] == [group_id]

    deleted, code = handle_speaker_request(
        {"command": "delete_group", "group_id": group_id},
        registry=registry,
        group_store=groups,
        bound_user_id="james",
    )
    assert code == 0
    assert deleted["deleted"] is True
    assert deleted["lifecycle"]["state"] == "verified"


def test_group_mutation_rejects_member_not_authorized_for_bound_user(tmp_path: Path) -> None:
    kitchen = _target("ha:media_player.kitchen", "Kitchen")
    bedroom = _target("ha:media_player.bedroom", "Bedroom")
    registry = _registry(kitchen, bedroom)
    groups = _groups(tmp_path, registry)

    body, code = handle_speaker_request(
        {
            "command": "create_group",
            "name": "Private Kitchen",
            "member_ids": [kitchen.id],
            "user": "james",
            "data_scope": "private",
        },
        registry=registry,
        group_store=groups,
        bound_user_id="cole",
    )

    assert code == 1
    assert body["ok"] is False
    assert "authorized" in body["error"].casefold()
    assert groups.list() == ()


def test_list_groups_hides_group_if_bound_user_cannot_see_every_member(tmp_path: Path) -> None:
    kitchen = _target("ha:media_player.kitchen", "Kitchen")
    bedroom = _target("ha:media_player.bedroom", "Bedroom")
    registry = _registry(kitchen, bedroom)
    groups = _groups(tmp_path, registry)
    group = groups.create("Whole House", [kitchen.id, bedroom.id])

    body, code = handle_speaker_request(
        {"command": "list_groups", "user": "james"},
        registry=registry,
        group_store=groups,
        bound_user_id="cole",
    )

    assert code == 0
    assert group.id not in json.dumps(body)
    assert body["groups"] == []


def test_bridge_main_executes_canonical_private_request(monkeypatch, tmp_path, capsys) -> None:
    import io

    import bridge.rex_speaker_bridge as speaker_bridge

    target = _target("ha:media_player.kitchen", "Kitchen")
    registry = _registry(target)
    groups = _groups(tmp_path, registry)
    main = getattr(speaker_bridge, "main", None)
    assert callable(main)

    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps({"command": "list_targets", "user": "james", "data_scope": "private"})
        ),
    )
    main(runtime_factory=lambda: (registry, groups, lambda: registry))
    body = json.loads(capsys.readouterr().out)
    assert body["ok"] is True
    assert [item["id"] for item in body["targets"]] == [target.id]
