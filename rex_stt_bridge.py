"""Persistent STT bridge — loads Whisper once and transcribes audio on demand.

Spawned by the Electron GUI main process (src/main/handlers/chat.ts) and kept
alive for the session so the Whisper model is never reloaded between requests.

Protocol (NDJSON over stdio):
  stdout after startup:
    {"type": "ready", "model": "<model_name>"}   — model loaded, ready for requests
    {"type": "error", "error": "<msg>"}           — fatal startup error (process exits)

  stdin per request (one JSON line):
    {"audio_base64": "<base64 WAV bytes>", "request_id": "<id>"}

  stdout per response (one JSON line):
    {"ok": true,  "transcript": "<text>",   "request_id": "<id>"}
    {"ok": false, "error":      "<msg>",    "request_id": "<id>"}
"""
from __future__ import annotations

import base64
import json
import sys
import tempfile
from pathlib import Path


def emit(obj: dict) -> None:  # type: ignore[type-arg]
    print(json.dumps(obj), flush=True)


def main() -> None:
    # Resolve whisper model name from Rex config (fall back to "base").
    model_name = "base"
    try:
        from rex.config import load_config  # type: ignore[import]

        cfg = load_config()
        model_name = getattr(cfg, "whisper_model", "base") or "base"
    except Exception:
        pass

    # Load Whisper model once on startup.
    try:
        import whisper  # type: ignore[import]
    except ImportError:
        emit({"type": "error", "error": "openai-whisper is not installed"})
        sys.exit(1)

    # Resolve device: honour config, auto-detect CUDA when set to "auto".
    device = "auto"
    try:
        from rex.config import load_config as _load_cfg

        _cfg = _load_cfg()
        device = getattr(_cfg, "whisper_device", "auto") or "auto"
    except Exception:
        pass
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    try:
        model = whisper.load_model(model_name, device=device)
    except Exception as exc:
        emit({"type": "error", "error": f"Failed to load Whisper '{model_name}': {exc}"})
        sys.exit(1)

    emit({"type": "ready", "model": model_name, "device": device})

    # Process transcription requests from stdin one line at a time.
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue

        request_id = ""
        try:
            payload = json.loads(raw)
            request_id = str(payload.get("request_id", ""))
            audio_b64 = str(payload.get("audio_base64", ""))

            if not audio_b64:
                emit({"ok": False, "error": "Missing audio_base64", "request_id": request_id})
                continue

            audio_bytes = base64.b64decode(audio_b64)

            # Write audio to a temporary WAV file and transcribe.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                result = model.transcribe(tmp_path, language="en", fp16=False)
                transcript = str(result.get("text", "")).strip()
                emit({"ok": True, "transcript": transcript, "request_id": request_id})
            finally:
                try:
                    Path(tmp_path).unlink()
                except OSError:
                    pass

        except Exception as exc:
            emit({"ok": False, "error": str(exc), "request_id": request_id})


if __name__ == "__main__":
    main()
