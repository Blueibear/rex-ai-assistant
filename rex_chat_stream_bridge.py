"""Streaming Rex chat bridge.

Reads JSON from stdin:  {"message": "<text>"}
Writes NDJSON to stdout, one JSON object per line:
  {"type": "token", "token": "<text>"}   – a chunk of the LLM response
  {"type": "done"}                        – stream complete (last line)
  {"type": "error", "error": "<text>"}   – error; process exits 1

Falls back to non-streaming (emits the full reply as a single "token" line)
when the backend's Assistant class does not expose generate_reply_stream.
"""

from __future__ import annotations

import asyncio
import json
import sys


def emit(obj: dict) -> None:  # noqa: ANN001
    print(json.dumps(obj), flush=True)


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        message = str(payload.get("message", ""))
    except Exception as exc:
        emit({"type": "error", "error": f"Bad input: {exc}"})
        sys.exit(1)

    async def run() -> None:
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
            # Prefer stream_reply (async generator) for true token-by-token streaming.
            # Fall back to generate_reply_stream for forward-compat, then to full reply.
            stream_fn = getattr(assistant, "stream_reply", None) or getattr(
                assistant, "generate_reply_stream", None
            )
            if stream_fn is not None:
                async for token in stream_fn(message):
                    emit({"type": "token", "token": str(token)})
            else:
                reply = await assistant.generate_reply(message)
                emit({"type": "token", "token": str(reply)})
        finally:
            shutdown_plugins(plugin_specs)

    try:
        asyncio.run(run())
        emit({"type": "done"})
    except Exception as exc:
        emit({"type": "error", "error": str(exc)})
        sys.exit(1)


if __name__ == "__main__":
    main()
