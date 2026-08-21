"""Authenticated per-user output-routing bridge for Electron settings."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from rex.identity import validate_user_id
from rex.media.accounts import MediaAccountStore
from rex.media.models import AudioTarget
from rex.media.registry import AudioTargetRegistry
from rex.output_routing.models import UserOutputPolicy
from rex.output_routing.service import (
    OutputRoutingService,
    _policy_from_dict,
    _policy_to_dict,
)

_POLICY_TARGET_FIELDS = (
    "spoken_response_target_id",
    "timer_target_id",
    "alarm_target_id",
    "media_target_id",
    "spoken_response_fallback_target_id",
    "timer_fallback_target_id",
    "alarm_fallback_target_id",
    "media_fallback_target_id",
)


def _authorized_target(registry: AudioTargetRegistry, target: AudioTarget, user_id: str) -> bool:
    resolution = registry.resolve(target.id, user_id=user_id)
    return resolution.target is not None or resolution.reason == "offline"


def _visible_targets(registry: AudioTargetRegistry, user_id: str) -> tuple[AudioTarget, ...]:
    return tuple(target for target in registry.targets if _authorized_target(registry, target, user_id))


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


def _serialize_account(account: Any) -> dict[str, str]:
    return {
        "provider": account.provider,
        "account_id": account.account_id,
        "display_name": account.display_name,
    }


def _bound_user(payload: Mapping[str, Any], bound_user_id: str | None) -> str:
    if bound_user_id is not None:
        owner = validate_user_id(bound_user_id)
        requested = payload.get("user_id") or payload.get("user")
        if requested is not None and str(requested) != owner:
            raise PermissionError("Routing policy is user-bound and cannot be edited for another user")
        return owner
    if payload.get("data_scope") != "private":
        raise PermissionError("private data scope is required")
    return validate_user_id(str(payload.get("user") or ""))


def _validate_policy_targets(
    policy: UserOutputPolicy,
    *,
    registry: AudioTargetRegistry,
    user_id: str,
) -> None:
    visible_ids = {target.id for target in _visible_targets(registry, user_id)}
    target_ids = [getattr(policy, name) for name in _POLICY_TARGET_FIELDS]
    for rule in policy.rules:
        target_ids.append(rule.target_id)
        target_ids.append(rule.fallback_target_id)
    unauthorized = sorted({target_id for target_id in target_ids if target_id and target_id not in visible_ids})
    if unauthorized:
        raise PermissionError("Routing policy contains an unauthorized audio target")


def _validate_default_account(
    policy: UserOutputPolicy,
    *,
    media_accounts: MediaAccountStore,
    user_id: str,
) -> None:
    if policy.default_media_provider is None:
        return
    account = media_accounts.get(
        user_id,
        policy.default_media_provider,
        policy.default_media_account_id or "",
    )
    if account is None:
        raise PermissionError("Default media account must belong to the active user")


async def _test_ha_target(target_id: str) -> bool:
    from rex.ha_tts.client import build_ha_tts_client

    client = build_ha_tts_client()
    if client is None:
        return False
    result = client.speak(
        "Rex audio routing test.",
        entity_id=target_id.split(":", 1)[1],
    )
    if hasattr(result, "__await__"):
        result = await result
    return bool(getattr(result, "ok", False))


def handle_output_routing_request(
    payload: Mapping[str, Any],
    *,
    registry: AudioTargetRegistry,
    routing: OutputRoutingService,
    media_accounts: MediaAccountStore,
    bound_user_id: str | None = None,
) -> tuple[dict[str, Any], int]:
    """Handle one user-bound routing settings request."""
    try:
        user_id = _bound_user(payload, bound_user_id)
        command = str(payload.get("command") or "").strip()

        if command == "get_policy":
            return {"ok": True, "policy": _policy_to_dict(routing.get_policy(user_id))}, 0

        if command == "update_policy":
            raw_policy = payload.get("policy")
            if not isinstance(raw_policy, dict):
                raise ValueError("policy must be an object")
            normalized_policy = dict(raw_policy)
            normalized_policy.setdefault("rules", [])
            policy = _policy_from_dict(normalized_policy)
            _validate_policy_targets(policy, registry=registry, user_id=user_id)
            _validate_default_account(policy, media_accounts=media_accounts, user_id=user_id)
            saved = routing.save_policy(user_id, policy)
            return {"ok": True, "policy": _policy_to_dict(saved)}, 0

        if command == "list_targets":
            targets = _visible_targets(registry, user_id)
            return {"ok": True, "targets": [_serialize_target(target) for target in targets]}, 0

        if command == "list_media_accounts":
            accounts = media_accounts.list(user_id)
            return {"ok": True, "accounts": [_serialize_account(account) for account in accounts]}, 0

        if command == "set_default_media_account":
            provider = payload.get("provider")
            account_id = payload.get("account_id")
            current = routing.get_policy(user_id)
            if provider is None and account_id is None:
                updated = replace(
                    current,
                    default_media_provider=None,
                    default_media_account_id=None,
                )
            else:
                if not isinstance(provider, str) or not provider.strip():
                    raise ValueError("provider is required")
                if not isinstance(account_id, str) or not account_id.strip():
                    raise ValueError("account_id is required")
                account = media_accounts.get(user_id, provider.strip(), account_id.strip())
                if account is None:
                    raise PermissionError("Media account must belong to the active user")
                updated = replace(
                    current,
                    default_media_provider=account.provider,
                    default_media_account_id=account.account_id,
                )
            routing.save_policy(user_id, updated)
            return {"ok": True, "policy": _policy_to_dict(updated)}, 0

        if command == "test_playback":
            target_id = payload.get("target_id")
            if not isinstance(target_id, str) or not target_id:
                raise ValueError("target_id is required")
            visible = {target.id: target for target in _visible_targets(registry, user_id)}
            target = visible.get(target_id)
            if target is None:
                raise PermissionError("Audio target is not authorized for this user")
            if not target.online:
                return {"ok": False, "error": "Audio target is offline."}, 1
            if target_id.startswith("ha:"):
                delivered = asyncio.run(_test_ha_target(target_id))
                return (
                    {"ok": True, "target_id": target_id}
                    if delivered
                    else {"ok": False, "error": "Test playback could not be verified."}
                ), (0 if delivered else 1)
            return {
                "ok": False,
                "error": "Test playback is not available for this target provider yet.",
            }, 1

        raise ValueError(f"unknown command: {command}")
    except (KeyError, PermissionError, TypeError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}, 1


def _runtime() -> tuple[AudioTargetRegistry, OutputRoutingService, MediaAccountStore]:
    from bridge.rex_speaker_bridge import _build_speaker_runtime

    registry, _group_store, _refresh = _build_speaker_runtime()
    return registry, OutputRoutingService(registry), MediaAccountStore()


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, dict):
            raise ValueError("request must be a JSON object")
        registry, routing, accounts = _runtime()
        body, _code = handle_output_routing_request(
            payload,
            registry=registry,
            routing=routing,
            media_accounts=accounts,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        body = {"ok": False, "error": "Output-routing request is invalid."}
    except Exception:
        body = {"ok": False, "error": "Output-routing service is unavailable."}
    print(json.dumps(body))


if __name__ == "__main__":
    main()
