"""Resolve the immutable user identity for one Electron main-process session."""

from __future__ import annotations

import json
import sys

from rex.bridge_utils import bridge_error_response


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        if payload.get("action") != "resolve_electron_session":
            raise ValueError("Unsupported identity action")

        from rex.config_manager import load_config
        from rex.identity import resolve_active_user, validate_user_id

        user_id = resolve_active_user(config=load_config())
        if user_id is None:
            raise PermissionError(
                "Electron has no active user. Select one with "
                "'rex identify --user <id>' before launching AskRex."
            )

        print(
            json.dumps(
                {
                    "ok": True,
                    "user_id": validate_user_id(user_id),
                    "authentication": "local-os-session",
                }
            ),
            flush=True,
        )
    except Exception as exc:
        print(json.dumps(bridge_error_response(exc)), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
