"""Hardened Flask API for text-to-speech synthesis."""

from __future__ import annotations

import os
import uuid
from contextlib import suppress

from flask import Flask, Response, jsonify, request, send_file
try:  # pragma: no cover - optional dependency
    from flask_cors import CORS
except ImportError:  # pragma: no cover - provide a tiny shim
    def CORS(app: Flask, **_kwargs):
        return app

try:  # pragma: no cover - optional dependency
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
except ImportError:  # pragma: no cover - lightweight stand-ins
    class Limiter:  # type: ignore[override]
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


def _create_app() -> Flask:
    app = Flask(__name__)
    limiter = Limiter(get_remote_address, app=app, default_limits=["30 per minute"])
    CORS(app, resources={r"/*": {"origins": "*"}})

    USERS_MAP = load_users_map()
    USER_PROFILES = load_all_profiles()
    default_user = resolve_user_key(os.getenv("REX_ACTIVE_USER"), USERS_MAP, profiles=USER_PROFILES)

    if not default_user:
        default_user = sorted(USER_PROFILES.keys())[0] if USER_PROFILES else "james"

    user_voices = {
        user: extract_voice_reference(profile)
        for user, profile in USER_PROFILES.items()
    }
    if default_user not in user_voices:
        user_voices[default_user] = None

    xtts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)

    required_api_key = os.getenv("REX_SPEAK_API_KEY")

    @app.route("/speak", methods=["POST"])
    @limiter.limit("15 per minute")
    def speak() -> Response:
        if required_api_key:
            supplied = request.headers.get("X-API-Key")
            if supplied != required_api_key:
                return jsonify({"error": "Invalid API key"}), 401

        try:
            payload = request.get_json(force=True)
        except BadRequest:
            return jsonify({"error": "Request body must be JSON"}), 400

        if not isinstance(payload, dict):
            return jsonify({"error": "Invalid payload"}), 400

        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Missing text parameter"}), 400

        user_param = payload.get("user")
        user = str(user_param).lower() if isinstance(user_param, str) else default_user
        if user not in user_voices:
            user = default_user

        speaker_wav = user_voices.get(user)
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
        except Exception as exc:  # pragma: no cover - synthesis issues
            return jsonify({"error": str(exc)}), 500
        finally:
            with suppress(FileNotFoundError):
                os.remove(output_path)

    return app


app = _create_app()
