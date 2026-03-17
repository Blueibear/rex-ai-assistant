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
from contextlib import suppress


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
    import rex
    from rex import settings as rex_settings
    from rex.assistant import Assistant
    from rex.config import load_config as load_runtime_config
    from rex.logging_utils import configure_logging
    from rex.plugins import load_plugins
    from rex.services import initialize_services
    from rex.voice_loop import (
        AsyncMicrophone,
        SpeechToText,
        TextToSpeech,
        VoiceLoop,
        WakeAcknowledgement,
    )
    from rex.wakeword.listener import build_default_detector

    configure_logging()
    initialize_services()

    try:
        runtime_config = load_runtime_config(reload=True)
        rex.settings = runtime_config
    except Exception:
        pass  # use whatever settings are already loaded

    # Read voice settings from config with sensible defaults
    sample_rate: int = 16000
    detection_seconds: float = 1.0
    capture_seconds: float = 5.0
    whisper_model: str = getattr(rex_settings, "whisper_model", "base") or "base"
    device: str = "cpu"
    language: str = "en"

    plugin_specs = load_plugins()
    assistant = Assistant(history_limit=rex_settings.max_memory_items, plugins=plugin_specs)

    mic = AsyncMicrophone(
        sample_rate=sample_rate,
        detection_seconds=detection_seconds,
        capture_seconds=capture_seconds,
    )

    wake_listener = build_default_detector(
        sample_rate=sample_rate,
        chunk_duration=detection_seconds,
    )

    stt = SpeechToText(model_name=whisper_model, device=device)
    tts = TextToSpeech(language=language)
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
        return text

    # Wrap speak: emit speaking state + rex transcript, then restore listening
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
        # Signal that we're ready to listen for the next wake word
        emit({"type": "state", "state": "listening"})

    voice_loop = VoiceLoop(
        assistant,
        wake_listener=wake_listener,
        detection_source=mic.detection_frame,
        record_phrase=mic.record_phrase,
        transcribe=wrapped_transcribe,
        speak=wrapped_speak,
        acknowledge=ack.play,
    )

    # Announce listening state now that the pipeline is ready
    emit({"type": "state", "state": "listening"})

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
    # Start stdin watcher so stop commands are handled immediately.
    watcher = threading.Thread(target=_stdin_watcher, daemon=True)
    watcher.start()

    # Announce initial idle state so the renderer knows the bridge is ready.
    emit({"type": "state", "state": "idle"})

    # Try to use the real voice loop; fall back to the stub simulation.
    try:
        asyncio.run(_run_real_loop())
    except ImportError as exc:
        # Voice dependencies (whisper, sounddevice, openWakeWord, etc.) missing
        emit({"type": "error", "error": f"Voice dependencies unavailable: {exc}"})
        _run_stub_loop()
    except Exception as exc:
        emit({"type": "error", "error": str(exc)})
        _run_stub_loop()


if __name__ == "__main__":
    main()
