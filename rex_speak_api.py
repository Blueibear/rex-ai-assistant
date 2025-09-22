"""Flask API that exposes the text-to-speech pipeline."""

from __future__ import annotations

import os
import secrets
import uuid
from contextlib import suppress
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from TTS.api import TTS

from assistant_errors import AuthenticationError, TextToSpeechError
from rex import settings
from rex.logging_utils import configure_logger
from rex.memory import extract_voice_reference, load_all_profiles, load_users_map, resolve_user_key

LOGGER = configure_logger(__name__)

app = Flask(__name__)
limiter = Limiter(get_remote_address, app=app, default_limits=[settings.flask_rate_limit])
CORS(app, resources={r"/*": {"origins": settings.allowed_origins}})

USERS_MAP = load_users_map()
USER_PROFILES = load_all_profiles()
DEFAULT_USER = resolve_user_key(settings.user_id, USERS_MAP, profiles=USER_PROFILES) or settings.user_id
USER_VOICES = {user: extract_voice_reference(profile) for user, profile in USER_PROFILES.items()}
USER_VOICES.setdefault(DEFAULT_USER, None)

# Load XTTS model at app startup with defensive logging.
try:  # pragma: no cover - heavy model load
    XTTS = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)
except Exception as exc:  # pragma: no cover - hardware specific
    LOGGER.error("Failed to load XTTS model: %s", exc)
    XTTS = None


def _require_api_key() -> None:
    expected = os.getenv("REX_SPEAK_API_KEY") or settings.speak_api_key
    if not expected:
        return
    provided = request.headers.get("X-API-Key") or request.headers.get("Authorization")
    if not provided or not secrets.compare_digest(provided.strip(), expected.strip()):
        raise AuthenticationError("Missing or invalid API key")


@app.errorhandler(AuthenticationError)
def _handle_auth_error(exc: AuthenticationError):  # pragma: no cover - Flask wiring
    LOGGER.warning("Authentication failure: %s", exc)
    return jsonify({"error": str(exc)}), 401


@app.errorhandler(TextToSpeechError)
def _handle_tts_error(exc: TextToSpeechError):  # pragma: no cover - Flask wiring
    LOGGER.error("TTS failure: %s", exc)
    return jsonify({"error": str(exc)}), 500


@app.route("/speak", methods=["POST"])
@limiter.limit(lambda: settings.flask_rate_limit)
def speak():
    _require_api_key()
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    user_param = payload.get("user")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "'text' must be a non-empty string"}), 400
    text = text.strip()
    if len(text) > 1_000:
        return jsonify({"error": "Text input exceeds 1000 characters"}), 400

    if user_param:
        user = str(user_param).lower()
    else:
        user = DEFAULT_USER

    if user not in USER_VOICES:
        user = DEFAULT_USER

    speaker_wav = USER_VOICES.get(user)
    if speaker_wav and not Path(speaker_wav).expanduser().is_file():
        return jsonify({"error": f"Speaker reference file not found for user '{user}'"}), 404

    if XTTS is None:
        raise TextToSpeechError("XTTS model unavailable")

    output_filename = f"rex_response_{uuid.uuid4().hex}.wav"
    output_path = Path(__file__).resolve().parent / output_filename

    try:
        XTTS.tts_to_file(text=text, speaker_wav=speaker_wav, language="en", file_path=str(output_path))
        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="rex_response.wav",
        )
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise TextToSpeechError(str(exc))
    finally:
        with suppress(FileNotFoundError):
            output_path.unlink()


__all__ = ["app"]
