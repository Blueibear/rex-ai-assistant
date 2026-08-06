"""Electron bridge for user profile operations.

Reads a JSON payload from stdin and writes a single JSON response to stdout.

Supported actions:
  - ``get``               -> return the immutable session user's composed profile view
  - ``update_preferences``-> accept preferences as a JSON object; update only the session user
  - ``set_avatar``        -> accept mime_type and strict base64 avatar_base64; update session user
  - ``remove_avatar``     -> remove the session user's avatar
"""

from __future__ import annotations

import base64
import json
import sys

from rex.bridge_utils import bridge_safe_error_response


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
    except Exception:
        print(json.dumps({"ok": False, "error": "Bad input"}), flush=True)
        sys.exit(1)

    if not isinstance(payload, dict):
        print(json.dumps({"ok": False, "error": "Bad input"}), flush=True)
        sys.exit(1)

    action = str(payload.get("action", "")).strip()

    try:
        import dataclasses

        from rex.identity import validate_user_id
        from rex.user_profile_service import UserProfileService

        # Validate session user and scope
        session_user_id = validate_user_id(str(payload.get("user") or ""))
        if payload.get("data_scope") != "private":
            raise PermissionError("Profile operations require private Electron data scope")

        service = UserProfileService()

        if action == "get":
            profile_view = service.get_profile(session_user_id)
            profile_dict = dataclasses.asdict(profile_view)
            print(json.dumps({"ok": True, "profile": profile_dict}), flush=True)
            return

        if action == "update_preferences":
            prefs = payload.get("preferences")
            if not isinstance(prefs, dict):
                raise ValueError("preferences must be a JSON object")
            service.update_preferences(session_user_id, prefs)
            print(json.dumps({"ok": True}), flush=True)
            return

        if action == "set_avatar":
            mime_type = str(payload.get("mime_type", "")).strip()
            avatar_b64 = str(payload.get("avatar_base64", "")).strip()

            if not mime_type:
                raise ValueError("mime_type is required")
            if not avatar_b64:
                raise ValueError("avatar_base64 is required")

            # Strict size validation before decoding
            if len(avatar_b64) > 2_900_000:  # 2.9 MiB encoded
                raise ValueError("Avatar data is too large")

            try:
                avatar_bytes = base64.b64decode(avatar_b64, validate=True)
            except Exception as exc:
                raise ValueError("Invalid base64 encoding") from exc

            service.set_avatar(session_user_id, avatar_bytes, mime_type)
            print(json.dumps({"ok": True}), flush=True)
            return

        if action == "remove_avatar":
            service.remove_avatar(session_user_id)
            print(json.dumps({"ok": True}), flush=True)
            return

        raise ValueError(f"Unsupported action: {action}")
    except Exception as exc:
        error_messages = {
            ValueError: "Request validation failed",
            PermissionError: "Permission denied",
            RuntimeError: "Request failed",
        }
        print(json.dumps(bridge_safe_error_response(exc, messages=error_messages)), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
