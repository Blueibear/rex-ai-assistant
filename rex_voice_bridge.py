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
import traceback as _traceback
from contextlib import suppress

from rex.bridge_utils import repo_root, resolve_python

_PYTHON_EXE = resolve_python()  # venv-aware interpreter path for subprocess calls
_REPO_ROOT = repo_root()  # absolute repo root for resolving scripts and config


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


async def _run_real_loop() -> None:
    """Build and run the real voice pipeline with state/transcript emission.

    Constructs VoiceLoop with the correct required arguments (assistant,
    wake_listener, detection_source, record_phrase, transcribe, speak) and
    wraps transcribe/speak to emit NDJSON events to stdout.
    """
    emit({"type": "status", "status": "importing_rex"})
    import rex
    from rex import settings as rex_settings
    emit({"type": "status", "status": "importing_assistant"})
    from rex.assistant import Assistant
    from rex import config as rex_config_module
    from rex.config import load_config as load_runtime_config
    from rex.logging_utils import configure_logging
    from rex.plugins import load_plugins
    from rex.services import initialize_services
    emit({"type": "status", "status": "importing_voice_loop"})
    import rex.voice_loop as voice_loop_module
    from rex.voice_loop import (
        AsyncMicrophone,
        SpeechToText,
        TextToSpeech,
        VoiceLoop,
        WakeAcknowledgement,
    )
    emit({"type": "status", "status": "importing_wakeword_detector"})
    from rex.wakeword.listener import build_default_detector

    configure_logging()
    emit({"type": "status", "status": "loading_config"})

    try:
        runtime_config = load_runtime_config(reload=True)
        active_settings = runtime_config
    except Exception:
        active_settings = rex_settings

    rex.settings = active_settings
    rex_config_module.settings = active_settings
    voice_loop_module.settings = active_settings
    configure_logging()
    emit({"type": "status", "status": "initializing_services"})
    initialize_services()

    # Read voice settings from config with sensible defaults
    sample_rate = int(getattr(active_settings, "sample_rate", 16000) or 16000)
    detection_seconds = float(
        getattr(active_settings, "detection_frame_seconds", 1.0) or 1.0
    )
    capture_seconds = float(
        getattr(active_settings, "capture_seconds", None)
        or getattr(active_settings, "command_duration", 5.0)
        or 5.0
    )
    whisper_model = str(getattr(active_settings, "whisper_model", "base") or "base")
    device = str(getattr(active_settings, "whisper_device", "auto") or "auto")
    language = str(getattr(active_settings, "whisper_language", "en") or "en")

    emit({"type": "status", "status": "loading_plugins"})
    plugin_specs = load_plugins()
    emit({"type": "status", "status": "creating_assistant"})
    assistant = Assistant(history_limit=active_settings.max_memory_items, plugins=plugin_specs)

    emit({"type": "status", "status": "initializing_microphone"})
    mic = AsyncMicrophone(
        sample_rate=sample_rate,
        detection_seconds=detection_seconds,
        capture_seconds=capture_seconds,
    )

    emit({"type": "status", "status": "loading_wakeword_detector"})
    wake_listener = build_default_detector(
        sample_rate=sample_rate,
        chunk_duration=detection_seconds,
        threshold=getattr(active_settings, "wakeword_threshold", 0.1),
        poll_interval=getattr(active_settings, "wakeword_poll_interval", 0.01),
        keyword=getattr(active_settings, "wakeword_keyword", None)
        or getattr(active_settings, "wakeword", None),
        model_path=getattr(active_settings, "wakeword_model_path", None),
        embedding_path=getattr(active_settings, "wakeword_embedding_path", None),
        backend=getattr(active_settings, "wakeword_backend", None),
        fallback_to_builtin=getattr(active_settings, "wakeword_fallback_to_builtin", True),
        fallback_keyword=getattr(active_settings, "wakeword_fallback_keyword", "hey jarvis"),
    )

    emit({"type": "status", "status": "initializing_stt"})
    stt = SpeechToText(model_name=whisper_model, device=device, async_load=True)
    emit({"type": "status", "status": "initializing_tts"})
    tts = TextToSpeech(language=language)
    emit({"type": "status", "status": "initializing_acknowledgement"})
    ack = WakeAcknowledgement()

    # Wrap transcribe: emit processing state, then emit user transcript
    async def wrapped_transcribe(audio) -> str:  # type: ignore[type-arg]
        emit({"type": "state", "state": "processing"})
        text = await stt.transcribe(audio, sample_rate)
        if text:
            emit(
                {
                    "type": "transcript",
                    "text": text,
                    "role": "user",
                    "timestamp": time_ms(),
                }
            )
        else:
            emit({"type": "state", "state": "idle"})
        return text

    # Wrap speak: emit speaking state + rex transcript, then restore wake-listening idle.
    async def wrapped_speak(text: str) -> None:
        emit({"type": "state", "state": "speaking"})
        emit(
            {
                "type": "transcript",
                "text": text,
                "role": "rex",
                "timestamp": time_ms(),
            }
        )
        await tts.speak(text)
        emit({"type": "state", "state": "idle"})

    def emit_voice_loop_state(status: str) -> None:
        if status in {"idle", "done"}:
            emit({"type": "state", "state": "idle"})
        elif status == "listening":
            emit({"type": "state", "state": "listening"})
        elif status in {"thinking", "executing"}:
            emit({"type": "state", "state": "processing"})
        elif status == "error":
            emit({"type": "error", "error": "Voice pipeline error; see logs for details."})
            emit({"type": "state", "state": "idle"})

    voice_loop = VoiceLoop(
        assistant,
        wake_listener=wake_listener,
        detection_source=mic.detection_frame,
        record_phrase=mic.record_phrase,
        transcribe=wrapped_transcribe,
        speak=wrapped_speak,
        acknowledge=ack.play,
        state_callback=emit_voice_loop_state,
    )

    # Announce readiness now that the real wake-word pipeline is built.
    emit({"type": "ready", "mode": "wake_word"})
    emit({"type": "state", "state": "idle"})

    # Start stdin watcher after startup. On Windows, reading a live stdin pipe
    # in a background thread during heavy imports can stall bridge readiness.
    watcher = threading.Thread(target=_stdin_watcher, daemon=True)
    watcher.start()

    # Run the voice loop as a cancellable task
    loop_task = asyncio.create_task(voice_loop.run())

    # Cancel the voice loop task when stop_event fires
    async def _wait_for_stop() -> None:
        event_loop = asyncio.get_running_loop()
        await event_loop.run_in_executor(None, stop_event.wait)
        loop_task.cancel()

    stop_watcher = asyncio.create_task(_wait_for_stop())

    try:
        await loop_task
    except asyncio.CancelledError:
        pass
    finally:
        stop_watcher.cancel()
        with suppress(asyncio.CancelledError):
            await stop_watcher

    emit({"type": "state", "state": "idle"})


def main() -> None:
    # Try to use the real voice loop. GUI wake-word mode should never silently
    # simulate listening because that hides microphone/dependency failures.
    try:
        asyncio.run(_run_real_loop())
    except ImportError as exc:
        # Voice dependencies (whisper, sounddevice, openWakeWord, etc.) missing
        emit({"type": "error", "error": f"Voice dependencies unavailable: {exc}"})
        sys.exit(1)
    except Exception as exc:
        emit({"type": "error", "error": str(exc), "traceback": _traceback.format_exc()})
        sys.exit(1)


if __name__ == "__main__":
    main()
