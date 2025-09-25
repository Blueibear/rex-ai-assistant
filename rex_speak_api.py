"""Hardened Flask API for text-to-speech synthesis."""

from __future__ import annotations

import os
import secrets
import uuid
from contextlib import suppress

from flask import Flask, jsonify, request, send_file, Response
try:
    from flask_cors import CORS
except ImportError:
    def CORS(app: Flask, **_kwargs):
        return app

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
except ImportError:
    class Limiter:
        def __init__(self, *args, **kwargs):
            pass

        def limit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

    def get_remote_address() -> str:
        return "0.0.0.0"

from werkzeug.exceptions import BadRequest
from TTS.api import TTS

from memory_utils import (
    extract_voice_reference,
    load_all_profiles,
    load_users_map,
    resolve_user_key,
)
from rex.config import settings
from rex.assistant_errors import AuthenticationError, TextToSpeechError

# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------

app = Flask(__name__)
limiter = Limiter(get_remote_address, app=app, default_limits=["30 per minute"])
CORS(app, resources={r"/*": {"origins": "*"}})

USERS_MAP = load_users_map()
USER_PROFILES = load_all_profiles()
DEFAULT_USER = resolve_user_key(os.getenv("REX_ACTIVE_USER"), USERS_MAP, profiles=USER_PROFILES)

if not DEFAULT_USER:
    DEFAULT_USER = sorted(USER_PROFILES.keys())[0] if USER_PROFILES else "james"

USER_VOICES = {
    user: extract_voice_reference(profile)
    for user, profile in USER_PROFILES.items()
}
if DEFAULT_USER not in USER_VOICES:
    USER_VOICES[DEFAULT_USER] = None

xtts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)
REQUIRED_API_KEY = os.getenv("REX_SPEAK_API_KEY")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _require_api_key() -> None:
    if not REQUIRED_API_KEY:
        return
    provided = request.headers.get("X-API-Key") or request.headers.get("Authorization")
    if not provided or not secrets.compare_digest(provided.strip(), REQUIRED_API_KEY.strip()):
        raise AuthenticationError("Missing or invalid API key")


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.errorhandler(AuthenticationError)
def _handle_auth_error(exc: AuthenticationError):
    return jsonify({"error": str(exc)}), 401

@app.errorhandler(TextToSpeechError)
def _handle_tts_error(exc: TextToSpeechError):
    return jsonify({"error": str(exc)}), 500

@app.route("/speak", methods=["POST"])
@limiter.limit("15 per minute")
def speak() -> Response:
    _require_api_key()

    try:
        payload = request.get_json(force=True)
    except BadRequest:
        return jsonify({"error": "Request body must be JSON"}), 400

    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON payload"}), 400

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing 'text' parameter"}), 400

    user_param = payload.get("user")
    user = str(user_param).lower() if isinstance(user_param, str) else DEFAULT_USER
    if user not in USER_VOICES:
        user = DEFAULT_USER

    speaker_wav = USER_VOICES.get(user)
    if speaker_wav and not os.path.exists(speaker_wav):
        return jsonify({"error": f"Speaker reference file not found for user '{user}'"}), 404

    output_filename = f"rex_response_{uuid.uuid4().hex}.wav"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)

    try:
        xtts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=settings.speak_language,
            file_path=output_path,
        )
        return send_file(output_path, mimetype="audio/wav", as_attachment=True, download_name="rex_response.wav")
    except Exception as exc:
        raise TextToSpeechError(str(exc))
    finally:
        with suppress(FileNotFoundError):
            os.remove(output_path)

