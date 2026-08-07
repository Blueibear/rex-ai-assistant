"""Authenticated Electron bridge for the immutable desktop user profile."""

from __future__ import annotations

import base64
import binascii
import dataclasses
import json
import sys
from typing import Any

from rex.bridge_utils import bridge_safe_error_response
from rex.identity import validate_user_id
from rex.user_profile_service import UserProfileService

_MAX_ENCODED_AVATAR = 2_900_000
_AUTHORITY_FIELDS = ("user_id", "target_user", "target_user_id")
_SAFE_MESSAGES: dict[type[BaseException], str] = {
    ValueError: "Request validation failed",
    PermissionError: "Permission denied",
    RuntimeError: "Profile operation failed",
    OSError: "Profile operation failed",
}


def _validated_session_user(payload: dict[str, Any]) -> str:
    raw_user = payload.get("user")
    if not isinstance(raw_user, str):
        raise ValueError("Invalid user")
    user_id = validate_user_id(raw_user)
    if payload.get("data_scope") != "private":
        raise PermissionError("Private scope required")
    for field in _AUTHORITY_FIELDS:
        if field not in payload:
            continue
        requested = payload[field]
        if requested not in (None, "", user_id):
            raise PermissionError("Cross-user profile operation denied")
    return user_id


def _strict_avatar_bytes(payload: dict[str, Any]) -> tuple[bytes, str]:
    mime_type = payload.get("mime_type")
    encoded = payload.get("avatar_base64")
    if mime_type not in {"image/jpeg", "image/png"}:
        raise ValueError("Unsupported avatar MIME type")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("Avatar data is required")
    if len(encoded) > _MAX_ENCODED_AVATAR:
        raise ValueError("Avatar data is too large")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid avatar encoding") from exc
    if not decoded:
        raise ValueError("Avatar data is required")
    return decoded, mime_type


def _profile_dict(service: UserProfileService, user_id: str) -> dict[str, Any]:
    return dataclasses.asdict(service.get_profile(user_id))


def process_payload(
    payload: object, *, service: UserProfileService | None = None
) -> tuple[dict[str, Any], int]:
    """Execute one profile request and return a renderer-safe response and exit code."""
    try:
        if not isinstance(payload, dict):
            raise ValueError("Payload must be an object")
        action = payload.get("action")
        if not isinstance(action, str) or not action:
            raise ValueError("Action is required")
        user_id = _validated_session_user(payload)
        profile_service = service or UserProfileService()

        if action == "get":
            pass
        elif action == "update_preferences":
            preferences = payload.get("preferences")
            if not isinstance(preferences, dict):
                raise ValueError("Preferences must be an object")
            profile_service.update_preferences(user_id, preferences)
        elif action == "set_avatar":
            avatar_bytes, mime_type = _strict_avatar_bytes(payload)
            profile_service.set_avatar(user_id, avatar_bytes, mime_type)
        elif action == "remove_avatar":
            profile_service.remove_avatar(user_id)
        else:
            raise ValueError("Unsupported action")

        return {"ok": True, "profile": _profile_dict(profile_service, user_id)}, 0
    except Exception as exc:
        return (
            bridge_safe_error_response(
                exc, messages=_SAFE_MESSAGES, default="Profile operation failed"
            ),
            1,
        )


def main() -> None:
    try:
        payload: object = json.loads(sys.stdin.read())
    except Exception as exc:
        response = bridge_safe_error_response(
            exc,
            messages={ValueError: "Request validation failed"},
            default="Request validation failed",
        )
        print(json.dumps(response), flush=True)
        raise SystemExit(1) from None

    response, exit_code = process_payload(payload)
    print(json.dumps(response), flush=True)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
