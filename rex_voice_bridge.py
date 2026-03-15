"""Rex voice bridge — persistent NDJSON event emitter for the voice panel.

Spawned by the Electron GUI main process (src/main/handlers/voice.ts).

Emits NDJSON lines to stdout:
  {"type": "state",      "state": "idle"|"listening"|"processing"|"speaking"}
  {"type": "transcript", "text": "...", "role": "user"|"rex", "timestamp": <ms>}
  {"type": "error",      "error": "..."}

Reads control commands from stdin (one JSON object per line):
  {"command": "stop"}
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time


def emit(obj: dict) -> None:  # type: ignore[type-arg]
    print(json.dumps(obj), flush=True)


def time_ms() -> int:
    return int(time.time() * 1000)


stop_event = threading.Event()


def _stdin_watcher() -> None:
    """Background thread: watch stdin for a stop command."""
    try:
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                cmd = json.loads(raw)
                if cmd.get("command") == "stop":
                    stop_event.set()
                    break
            except Exception:
                pass
    except Exception:
        pass
    # stdin closed or stop received
    stop_event.set()


def _run_stub_loop() -> None:
    """Simulate voice sessions until a stop command is received.

    Used when the real voice pipeline is unavailable.  Attempts a real LLM
    reply for the stub 'user turn'; falls back to a canned response if the
    backend is not importable.
    """
    try:
        from rex import settings as rex_settings  # type: ignore[import]
        from rex.assistant import Assistant  # type: ignore[import]
        from rex.services import initialize_services  # type: ignore[import]

        initialize_services()
        assistant = Assistant(history_limit=rex_settings.max_memory_items, plugins=[])
        has_backend = True
    except Exception:
        has_backend = False
        assistant = None  # type: ignore[assignment]

    while not stop_event.is_set():
        # ── Listening phase ──────────────────────────────────────────────────
        emit({"type": "state", "state": "listening"})
        if stop_event.wait(timeout=4.0):
            break

        # ── Processing phase ─────────────────────────────────────────────────
        emit({"type": "state", "state": "processing"})
        stub_user_text = "[Stub mode — microphone not active]"
        emit(
            {
                "type": "transcript",
                "text": stub_user_text,
                "role": "user",
                "timestamp": time_ms(),
            }
        )
        if stop_event.wait(timeout=0.5):
            break

        # ── LLM reply ────────────────────────────────────────────────────────
        if has_backend and assistant is not None:
            try:
                reply_text = str(asyncio.run(assistant.generate_reply(stub_user_text)))
            except Exception as exc:
                reply_text = f"(Backend error: {exc})"
        else:
            reply_text = (
                "Voice pipeline is running in stub mode. "
                "Install all voice dependencies and connect a microphone to enable real voice input."
            )

        if stop_event.is_set():
            break

        # ── Speaking phase ────────────────────────────────────────────────────
        emit({"type": "state", "state": "speaking"})
        emit(
            {
                "type": "transcript",
                "text": reply_text,
                "role": "rex",
                "timestamp": time_ms(),
            }
        )
        if stop_event.wait(timeout=2.5):
            break

        # ── Brief idle between sessions ───────────────────────────────────────
        emit({"type": "state", "state": "idle"})
        if stop_event.wait(timeout=1.0):
            break

    emit({"type": "state", "state": "idle"})


def main() -> None:
    # Start stdin watcher so stop commands are handled immediately.
    watcher = threading.Thread(target=_stdin_watcher, daemon=True)
    watcher.start()

    # Announce initial idle state so the renderer knows the bridge is ready.
    emit({"type": "state", "state": "idle"})

    # Try to use the real voice loop; fall back to the stub simulation.
    try:
        from rex.voice_loop import VoiceLoop  # type: ignore[import]
        from rex.logging_utils import configure_logging  # type: ignore[import]
        from rex.services import initialize_services  # type: ignore[import]

        configure_logging()
        initialize_services()

        loop = VoiceLoop(
            on_state_change=lambda s: emit({"type": "state", "state": s}),
            on_transcript=lambda text, role: emit(
                {"type": "transcript", "text": text, "role": role, "timestamp": time_ms()}
            ),
        )
        loop.run_until(stop_event)
    except (ImportError, AttributeError):
        # Real voice pipeline not available — run stub.
        _run_stub_loop()
    except Exception as exc:
        emit({"type": "error", "error": str(exc)})
        _run_stub_loop()


if __name__ == "__main__":
    main()
