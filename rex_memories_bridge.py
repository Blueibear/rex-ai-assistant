"""Rex memories bridge for Electron GUI.

Reads a JSON command from stdin and writes a JSON response to stdout.

Commands:
  {"command": "list"}
    -> {"ok": true, "memories": [...]}

  {"command": "add", "data": {text, category}}
    -> {"ok": true, "memory": {...}}

  {"command": "update", "id": "<entry_id>", "data": {text, category}}
    -> {"ok": true, "memory": {...}}

  {"command": "delete", "id": "<entry_id>"}
    -> {"ok": true}

Memory format (GUI):
  {id, text, category, createdAt (ISO), updatedAt (ISO)}
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _entry_to_gui(entry: Any) -> dict[str, Any]:
    """Convert a LongTermMemory MemoryEntry to the GUI Memory dict format."""
    content: dict[str, Any] = entry.content or {}
    text = str(content.get("text") or "")
    updated_at = content.get("updated_at") or None

    created_at_str: str
    if hasattr(entry.created_at, "isoformat"):
        created_at_str = entry.created_at.isoformat()
    else:
        created_at_str = str(entry.created_at)

    updated_at_str: str = updated_at if isinstance(updated_at, str) else created_at_str

    return {
        "id": entry.entry_id,
        "text": text,
        "category": entry.category,
        "createdAt": created_at_str,
        "updatedAt": updated_at_str,
    }


def _handle_list() -> dict[str, Any]:
    from rex.memory import get_long_term_memory  # type: ignore[import]

    ltm = get_long_term_memory()
    entries = ltm.search()
    return {"ok": True, "memories": [_entry_to_gui(e) for e in entries]}


def _handle_add(data: dict[str, Any]) -> dict[str, Any]:
    from rex.memory import get_long_term_memory  # type: ignore[import]

    text = str(data.get("text") or "").strip()
    category = str(data.get("category") or "general").strip() or "general"

    if not text:
        return {"ok": False, "error": "Memory text is required"}

    now = _utc_now_iso()
    ltm = get_long_term_memory()
    entry = ltm.add_entry(
        category=category,
        content={"text": text, "updated_at": now},
    )
    return {"ok": True, "memory": _entry_to_gui(entry)}


def _handle_update(entry_id: str, data: dict[str, Any]) -> dict[str, Any]:
    from rex.memory import get_long_term_memory  # type: ignore[import]

    text = str(data.get("text") or "").strip()
    category = str(data.get("category") or "").strip()

    if not text:
        return {"ok": False, "error": "Memory text is required"}

    ltm = get_long_term_memory()
    entry = ltm.get_entry(entry_id)
    if entry is None:
        return {"ok": False, "error": f"Memory {entry_id!r} not found"}

    entry.content["text"] = text
    entry.content["updated_at"] = _utc_now_iso()
    if category:
        entry.category = category
    ltm._save()  # noqa: SLF001

    return {"ok": True, "memory": _entry_to_gui(entry)}


def _handle_delete(entry_id: str) -> dict[str, Any]:
    from rex.memory import get_long_term_memory  # type: ignore[import]

    ltm = get_long_term_memory()
    deleted = ltm.forget(entry_id)
    if not deleted:
        return {"ok": False, "error": f"Memory {entry_id!r} not found"}
    return {"ok": True}


def main() -> None:
    try:
        payload: dict[str, Any] = json.loads(sys.stdin.read())
        command = str(payload.get("command") or "")
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Bad input: {exc}"}), flush=True)
        sys.exit(1)

    try:
        if command == "list":
            result = _handle_list()
        elif command == "add":
            result = _handle_add(dict(payload.get("data") or {}))
        elif command == "update":
            result = _handle_update(str(payload.get("id") or ""), dict(payload.get("data") or {}))
        elif command == "delete":
            result = _handle_delete(str(payload.get("id") or ""))
        else:
            result = {"ok": False, "error": f"Unknown command: {command!r}"}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
