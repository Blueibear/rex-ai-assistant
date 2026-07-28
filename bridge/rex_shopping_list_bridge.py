"""Shopping list IPC bridge for the Electron GUI.

Reads a JSON command from stdin, executes it against ShoppingList, and
writes a JSON result to stdout.

Supported commands:
  {"command": "list"}
  {"command": "add", "name": "...", "quantity": 1.0, "unit": ""}
  {"command": "check", "id": "..."}
  {"command": "uncheck", "id": "..."}
  {"command": "clear_checked"}
"""

from __future__ import annotations

import json
import sys

from rex.bridge_utils import bridge_error_response
from rex.shopping_list import ShoppingList


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        from rex.identity import validate_user_id

        actor_user_id = validate_user_id(str(payload.get("user") or ""))
        if payload.get("data_scope") != "shared_household":
            raise PermissionError("Shopping list requires explicit shared_household scope")
    except Exception as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": f"Bad input: {exc}"}))
        sys.exit(1)

    sl = ShoppingList()
    command = payload.get("action") or payload.get("command", "")

    try:
        if command == "list":
            items = sl.list_items()
            sys.stdout.write(json.dumps({"ok": True, "items": [i.to_dict() for i in items]}))

        elif command == "add":
            name = str(payload.get("name", "")).strip()
            if not name:
                sys.stdout.write(json.dumps({"ok": False, "error": "name is required"}))
                return
            quantity = float(payload.get("quantity", 1.0))
            unit = str(payload.get("unit", ""))
            item = sl.add_item(name, quantity=quantity, unit=unit, added_by=actor_user_id)
            sys.stdout.write(json.dumps({"ok": True, "item": item.to_dict()}))

        elif command == "check":
            item_id = str(payload.get("id", ""))
            found = sl.check_item(item_id)
            sys.stdout.write(
                json.dumps({"ok": found, "error": None if found else "item not found"})
            )

        elif command == "uncheck":
            item_id = str(payload.get("id", ""))
            found = sl.uncheck_item(item_id)
            sys.stdout.write(
                json.dumps({"ok": found, "error": None if found else "item not found"})
            )

        elif command == "clear_checked":
            count = sl.clear_checked()
            sys.stdout.write(json.dumps({"ok": True, "count": count}))

        else:
            sys.stdout.write(json.dumps({"ok": False, "error": f"unknown command: {command}"}))

    except Exception as exc:
        sys.stdout.write(json.dumps(bridge_error_response(exc)))


if __name__ == "__main__":
    main()
