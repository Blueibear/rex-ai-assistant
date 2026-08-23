"""Electron/adapter bridge for constitutional context and privacy settings."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from typing import Any

from rex.context.privacy import ContextPrivacyService, get_context_privacy_service
from rex.identity import validate_user_id


def _bound_user(payload: Mapping[str, Any], bound_user_id: str | None) -> str:
    if bound_user_id is not None:
        owner = validate_user_id(bound_user_id)
        requested = payload.get("user_id") or payload.get("user")
        if requested is not None and str(requested) != owner:
            raise PermissionError("Privacy settings cannot be changed for another user")
        return owner
    if payload.get("data_scope") != "private":
        raise PermissionError("private data scope is required")
    return validate_user_id(str(payload.get("user") or ""))


def _bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean")
    return value


def _handle_get_state(
    payload: Mapping[str, Any], service: ContextPrivacyService, user_id: str
) -> dict[str, Any]:
    del payload
    return {"ok": True, **service.get_state(user_id)}


def _handle_source_context(
    payload: Mapping[str, Any], service: ContextPrivacyService, user_id: str
) -> dict[str, Any]:
    source_id = payload.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id is required")
    source = service.set_source_context(
        owner_user_id=user_id,
        source_id=source_id,
        enabled=_bool(payload, "enabled"),
        actor_user_id=user_id,
    )
    return {"ok": True, "source": source}


def _handle_location_assist(
    payload: Mapping[str, Any], service: ContextPrivacyService, user_id: str
) -> dict[str, Any]:
    location = service.set_location_assist(
        owner_user_id=user_id,
        enabled=_bool(payload, "enabled"),
        actor_user_id=user_id,
    )
    return {"ok": True, "location": location}


def _handle_location_share(
    payload: Mapping[str, Any], service: ContextPrivacyService, user_id: str
) -> dict[str, Any]:
    recipient = payload.get("recipient_user_id")
    if not isinstance(recipient, str) or not recipient:
        raise ValueError("recipient_user_id is required")
    location = service.set_location_share(
        owner_user_id=user_id,
        recipient_user_id=recipient,
        enabled=_bool(payload, "enabled"),
        actor_user_id=user_id,
    )
    return {"ok": True, "location": location}


def _handle_upload_policy(
    payload: Mapping[str, Any], service: ContextPrivacyService, user_id: str
) -> dict[str, Any]:
    doc_id = payload.get("doc_id")
    audience = payload.get("audience_scope")
    if not isinstance(doc_id, str) or not doc_id:
        raise ValueError("doc_id is required")
    if audience not in {"private", "household"}:
        raise ValueError("audience_scope must be private or household")
    upload = service.update_upload_policy(
        owner_user_id=user_id,
        doc_id=doc_id,
        audience_scope=audience,
        context_enabled=_bool(payload, "context_enabled"),
        actor_user_id=user_id,
    )
    return {"ok": True, "upload": upload}


def _handle_proactive_assistance(
    payload: Mapping[str, Any], service: ContextPrivacyService, user_id: str
) -> dict[str, Any]:
    enabled = service.set_proactive_assistance(
        owner_user_id=user_id,
        enabled=_bool(payload, "enabled"),
        actor_user_id=user_id,
    )
    return {"ok": True, "proactive_assistance": enabled}


_CommandHandler = Callable[[Mapping[str, Any], ContextPrivacyService, str], dict[str, Any]]
_COMMAND_HANDLERS: dict[str, _CommandHandler] = {
    "get_state": _handle_get_state,
    "set_source_context": _handle_source_context,
    "set_location_assist": _handle_location_assist,
    "set_location_share": _handle_location_share,
    "update_upload_policy": _handle_upload_policy,
    "set_proactive_assistance": _handle_proactive_assistance,
}


def _dispatch(
    command: str,
    payload: Mapping[str, Any],
    *,
    service: ContextPrivacyService,
    user_id: str,
) -> dict[str, Any]:
    handler = _COMMAND_HANDLERS.get(command)
    if handler is None:
        raise ValueError("unsupported context privacy command")
    return handler(payload, service, user_id)


def handle_context_policy_request(
    payload: Mapping[str, Any],
    *,
    service: ContextPrivacyService | None = None,
    bound_user_id: str | None = None,
) -> tuple[dict[str, Any], int]:
    try:
        user_id = _bound_user(payload, bound_user_id)
        command = payload.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError("command is required")
        body = _dispatch(
            command,
            payload,
            service=service or get_context_privacy_service(),
            user_id=user_id,
        )
        return body, 0
    except PermissionError:
        return {"ok": False, "error": "Privacy settings are user-bound."}, 1
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}, 1


def main() -> int:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw or "{}")
        if not isinstance(payload, dict):
            raise ValueError("request must be an object")
        body, code = handle_context_policy_request(payload)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        body, code = {"ok": False, "error": str(exc)}, 1
    sys.stdout.write(json.dumps(body))
    sys.stdout.flush()
    return code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["handle_context_policy_request", "main"]
