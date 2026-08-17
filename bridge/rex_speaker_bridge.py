"""Authenticated canonical speaker/group bridge for the Electron GUI."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from uuid import uuid4

from rex.actions.lifecycle import ActionLifecycle, ActionState
from rex.bridge_utils import bridge_error_response
from rex.identity import validate_user_id
from rex.media.groups import SpeakerGroup, SpeakerGroupStore
from rex.media.models import AudioTarget, TargetKind, TargetProviderAdapter
from rex.media.registry import AudioTargetRegistry


def _target_authorized(registry: AudioTargetRegistry, target: AudioTarget, user_id: str) -> bool:
    resolution = registry.resolve(target.id, user_id=user_id)
    return resolution.target is not None or resolution.reason == "offline"


def _visible_targets(registry: AudioTargetRegistry, user_id: str) -> tuple[AudioTarget, ...]:
    return tuple(
        target for target in registry.targets if _target_authorized(registry, target, user_id)
    )


def _serialize_target(target: AudioTarget) -> dict[str, Any]:
    return {
        "id": target.id,
        "name": target.display_name,
        "provider": target.provider,
        "kind": target.kind.value,
        "room": target.room,
        "capabilities": sorted(capability.value for capability in target.capabilities),
        "online": target.online,
        "health": target.health,
    }


def _serialize_group(group: SpeakerGroup) -> dict[str, Any]:
    return {
        "id": group.id,
        "name": group.name,
        "member_ids": list(group.member_ids),
        # Member capability intersection is metadata, not proof of group-execution support.
        "capabilities": [],
    }


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def _required_members(payload: Mapping[str, Any]) -> tuple[str, ...]:
    raw = payload.get("member_ids")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("member_ids must be a list")
    members = tuple(raw)
    if not all(isinstance(member, str) and member for member in members):
        raise ValueError("member_ids must contain non-empty target IDs")
    return members


def _verified_group_mutation(
    command: str,
    mutate: Callable[[], Any],
    verify: Callable[[Any], bool],
) -> tuple[Any, dict[str, object]]:
    lifecycle = ActionLifecycle.create(
        action_id=f"speaker-group:{command}:{uuid4()}",
    )
    lifecycle.transition(ActionState.AUTHORIZED, evidence_ref="speaker-group:authorized")
    lifecycle.transition(ActionState.ATTEMPTED, evidence_ref="speaker-group:attempted")
    value = mutate()
    lifecycle.transition(ActionState.COMPLETED, evidence_ref="speaker-group:persisted")
    if verify(value):
        snapshot = lifecycle.transition(
            ActionState.VERIFIED, evidence_ref="speaker-group:reread-verified"
        )
    else:
        snapshot = lifecycle.transition(
            ActionState.UNVERIFIED, evidence_ref="speaker-group:reread-mismatch"
        )
    return value, snapshot.to_dict()


def handle_speaker_request(
    payload: Mapping[str, Any],
    *,
    registry: AudioTargetRegistry,
    group_store: SpeakerGroupStore,
    bound_user_id: str | None = None,
    refresh_registry: Callable[[], AudioTargetRegistry] | None = None,
) -> tuple[dict[str, Any], int]:
    """Handle one canonical speaker request with caller-bound user authority."""
    try:
        if bound_user_id is None:
            if payload.get("data_scope") != "private":
                raise ValueError("private data scope is required")
            user_id = validate_user_id(str(payload.get("user") or ""))
        else:
            user_id = validate_user_id(bound_user_id)

        command = str(payload.get("command") or payload.get("action") or "").strip()
        current_registry = registry
        if command == "refresh_targets":
            if refresh_registry is not None:
                current_registry = refresh_registry()
            targets = _visible_targets(current_registry, user_id)
            return {"ok": True, "targets": [_serialize_target(target) for target in targets]}, 0
        if command in {"list_targets", "list"}:
            targets = _visible_targets(current_registry, user_id)
            return {"ok": True, "targets": [_serialize_target(target) for target in targets]}, 0

        visible_ids = {target.id for target in _visible_targets(current_registry, user_id)}

        def ensure_members_authorized(member_ids: Sequence[str]) -> None:
            unauthorized = [member_id for member_id in member_ids if member_id not in visible_ids]
            if unauthorized:
                raise PermissionError("Speaker group members must be authorized audio targets")

        if command == "list_groups":
            groups = [
                group
                for group in group_store.list()
                if all(member_id in visible_ids for member_id in group.member_ids)
            ]
            return {"ok": True, "groups": [_serialize_group(group) for group in groups]}, 0
        if command == "create_group":
            members = _required_members(payload)
            ensure_members_authorized(members)
            group, lifecycle = _verified_group_mutation(
                command,
                lambda: group_store.create(_required_string(payload, "name"), members),
                lambda created: group_store.get(created.id) == created,
            )
            return {"ok": True, "group": _serialize_group(group), "lifecycle": lifecycle}, 0
        if command == "rename_group":
            group_id = _required_string(payload, "group_id")
            current = group_store.get(group_id)
            if current is None:
                raise KeyError("Unknown speaker group")
            ensure_members_authorized(current.member_ids)
            group, lifecycle = _verified_group_mutation(
                command,
                lambda: group_store.rename(group_id, _required_string(payload, "name")),
                lambda renamed: group_store.get(group_id) == renamed,
            )
            return {"ok": True, "group": _serialize_group(group), "lifecycle": lifecycle}, 0
        if command == "set_group_members":
            group_id = _required_string(payload, "group_id")
            current = group_store.get(group_id)
            if current is None:
                raise KeyError("Unknown speaker group")
            ensure_members_authorized(current.member_ids)
            members = _required_members(payload)
            ensure_members_authorized(members)
            group, lifecycle = _verified_group_mutation(
                command,
                lambda: group_store.set_members(group_id, members),
                lambda changed: group_store.get(group_id) == changed,
            )
            return {"ok": True, "group": _serialize_group(group), "lifecycle": lifecycle}, 0
        if command == "delete_group":
            group_id = _required_string(payload, "group_id")
            current = group_store.get(group_id)
            if current is None:
                return {"ok": True, "deleted": False}, 0
            ensure_members_authorized(current.member_ids)
            deleted, lifecycle = _verified_group_mutation(
                command,
                lambda: group_store.delete(group_id),
                lambda removed: bool(removed) and group_store.get(group_id) is None,
            )
            return {"ok": True, "deleted": deleted, "lifecycle": lifecycle}, 0
        raise ValueError(f"unknown command: {command}")
    except (KeyError, PermissionError, TypeError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}, 1


def _build_speaker_runtime() -> tuple[
    AudioTargetRegistry,
    SpeakerGroupStore,
    Callable[[], AudioTargetRegistry],
]:
    from rex.audio.speaker_discovery import get_speaker_discovery
    from rex.config import settings
    from rex.ha_bridge import HABridge
    from rex.identity import list_known_users
    from rex.media.adapters import HomeAssistantMediaAdapter, SmartSpeakerAdapter
    from rex.permissions import get_permissions

    adapters: list[TargetProviderAdapter] = [SmartSpeakerAdapter(get_speaker_discovery())]
    if settings.ha_base_url and settings.ha_token:
        try:
            adapters.append(HomeAssistantMediaAdapter(HABridge()))
        except Exception:
            pass

    def build_snapshot() -> tuple[AudioTargetRegistry, SpeakerGroupStore]:
        base_targets: list[AudioTarget] = []
        for adapter in adapters:
            try:
                base_targets.extend(adapter.discover_targets())
            except Exception:
                continue
        by_id = {target.id: target for target in base_targets}
        group_store = SpeakerGroupStore(
            target_exists=by_id.__contains__,
            target_capabilities=lambda target_id: by_id[target_id].capabilities,
        )
        try:
            groups = group_store.list()
        except (KeyError, ValueError):
            groups = ()
        targets = list(base_targets)
        for group in groups:
            members = tuple(by_id[member_id] for member_id in group.member_ids)
            online = bool(members) and all(member.online for member in members)
            targets.append(
                AudioTarget(
                    id=group.id,
                    native_id=group.id,
                    provider="group",
                    kind=TargetKind.GROUP,
                    display_name=group.name,
                    aliases=(),
                    room=None,
                    capabilities=frozenset(),
                    online=online,
                    health="configured" if online else "member_unavailable",
                )
            )

        known_users = {
            str(user["id"])
            for user in list_known_users()
            if isinstance(user, dict) and isinstance(user.get("id"), str)
        }
        base_ids = frozenset(by_id)
        group_members = {group.id: frozenset(group.member_ids) for group in groups}
        authorized: dict[str, frozenset[str]] = {}
        for user_id in known_users:
            try:
                permissions = set(get_permissions(user_id))
            except Exception:
                permissions = set()
            allowed_base = (
                base_ids if permissions.intersection({"ha_control", "admin"}) else frozenset()
            )
            allowed_groups = {
                group_id
                for group_id, member_ids in group_members.items()
                if member_ids and member_ids.issubset(allowed_base)
            }
            authorized[user_id] = frozenset((*allowed_base, *allowed_groups))

        return AudioTargetRegistry(targets, authorized_target_ids=authorized), group_store

    registry, group_store = build_snapshot()

    def refresh() -> AudioTargetRegistry:
        return build_snapshot()[0]

    return registry, group_store, refresh


def _legacy_list_speakers() -> dict[str, Any]:
    """Preserve the pre-US-121 discovery-only bridge contract.

    This path exposes only LAN discovery metadata and grants no canonical
    target/group authority. Authenticated Electron callers use the private
    canonical commands below.
    """
    try:
        from rex.audio.speaker_discovery import SpeakerDiscoveryService

        service = SpeakerDiscoveryService(
            refresh_interval_seconds=60.0,
            discovery_timeout_seconds=1.0,
        )
        speakers = service.discover_now()
        return {
            "ok": True,
            "speakers": [
                {
                    "provider": speaker.provider,
                    "name": speaker.name,
                    "ip": speaker.ip,
                    "model": speaker.model,
                }
                for speaker in speakers
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {**bridge_error_response(exc), "speakers": []}


def main(
    *,
    runtime_factory: Callable[
        [], tuple[AudioTargetRegistry, SpeakerGroupStore, Callable[[], AudioTargetRegistry]]
    ] = _build_speaker_runtime,
) -> None:
    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, dict):
            raise ValueError("request must be a JSON object")
        command = str(payload.get("command") or payload.get("action") or "list").strip()
        if command == "list" and payload.get("data_scope") != "private":
            print(json.dumps(_legacy_list_speakers()))
            return
        registry, group_store, refresh = runtime_factory()
        body, _code = handle_speaker_request(
            payload,
            registry=registry,
            group_store=group_store,
            refresh_registry=refresh,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        body = {"ok": False, "error": "Speaker request is invalid."}
    except Exception:
        body = {"ok": False, "error": "Speaker service is unavailable."}
    print(json.dumps(body))


if __name__ == "__main__":
    main()
