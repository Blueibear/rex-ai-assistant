"""Electron bridge for voice identity enrollment UI actions.

Reads a JSON payload from stdin and writes a single JSON response to stdout.

Supported actions:
  - ``list``   -> return active user and enrolled users
  - ``enroll`` -> store an embedding from three captured samples
  - ``delete`` -> remove an enrolled user
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Bad input: {exc}"}), flush=True)
        sys.exit(1)

    action = str(payload.get("action", "list"))

    try:
        from rex.voice_identity.ui_service import (
            delete_enrollment,
            enroll_from_samples,
            get_active_user_id,
            list_enrollments,
        )

        if action == "list":
            print(
                json.dumps(
                    {
                        "ok": True,
                        "active_user_id": get_active_user_id(),
                        "enrollments": list_enrollments(),
                    }
                ),
                flush=True,
            )
            return

        if action == "enroll":
            user_id = str(payload.get("user_id", "")).strip()
            samples = payload.get("samples", [])
            if not user_id:
                raise ValueError("user_id is required")
            if not isinstance(samples, list):
                raise ValueError("samples must be a list")
            enrollment = enroll_from_samples(user_id, samples)
            print(json.dumps({"ok": True, "enrollment": enrollment}), flush=True)
            return

        if action == "delete":
            user_id = str(payload.get("user_id", "")).strip()
            if not user_id:
                raise ValueError("user_id is required")
            deleted = delete_enrollment(user_id)
            print(json.dumps({"ok": True, "deleted": deleted}), flush=True)
            return

        raise ValueError(f"Unsupported action: {action}")
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
