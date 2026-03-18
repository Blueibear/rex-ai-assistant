"""Single-turn Rex chat bridge.

Reads a JSON payload from stdin: {"message": "<text>"}
Writes a JSON response to stdout: {"ok": true, "reply": "<text>"}
                               or {"ok": false, "error": "<text>"}

Used by the Electron GUI main process (src/main/handlers/chat.ts) to forward
one chat message to the Rex backend and return the response.
"""

from __future__ import annotations

import asyncio
import json
import sys


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        message = str(payload.get("message", ""))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Bad input: {exc}"}), flush=True)
        sys.exit(1)

    async def run() -> str:
        from rex import settings  # type: ignore[import]
        from rex.assistant import Assistant  # type: ignore[import]
        from rex.logging_utils import configure_logging  # type: ignore[import]
        from rex.plugins import load_plugins, shutdown_plugins  # type: ignore[import]
        from rex.services import initialize_services  # type: ignore[import]

        configure_logging()
        initialize_services()
        plugin_specs = load_plugins()
        assistant = Assistant(history_limit=settings.max_memory_items, plugins=plugin_specs)
        try:
            reply = await assistant.generate_reply(message)
            return str(reply)
        finally:
            shutdown_plugins(plugin_specs)

    try:
        reply = asyncio.run(run())
        print(json.dumps({"ok": True, "reply": reply}), flush=True)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
