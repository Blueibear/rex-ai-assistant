"""Hardened Flask API for text-to-speech synthesis."""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional, Tuple

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from rex.config import _parse_int
from rex.ha_bridge import create_blueprint as create_ha_blueprint

try:
    from flask_limiter import Limiter
    from flask_limiter.exceptions import RateLimitExceeded
except ImportError:
    class Limiter:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def limit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

    class RateLimitExceeded(Exception):  # type: ignore
        retry_after = None

from TTS.api import TTS
from rex.assistant_errors import AuthenticationError, TextToSpeechError
from rex.memory_utils import (
    extract_voice_reference,
    load_all_profiles,
    load_users_map,
    resolve_user_key,
)

# ------------------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------------------

app = Flask(__name__)

ALLOWED_ORIGINS = os.getenv(
    "REX_ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5000,http://127.0.0.1:3000,http://127.0.0.1:5000",
)
_CORS_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS.split(",") if origin.strip()]

CORS(
    app,
    resources={
        r"/*": {
            "origins": _CORS_ORIGINS,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-API-Key"],
            "expose_headers": ["Retry-After"],
            "supports_credentials": False,
            "max_age": 600,
        }
    },
)

logger = logging.getLogger("rex.speak_api")

if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("REX_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ------------------------------------------------------------------------------
# Rate Limiting
# ------------------------------------------------------------------------------

TRUSTED_PROXIES = set(
    ip.strip()
    for ip in os.getenv("REX_TRUSTED_PROXIES", "127.0.0.1,::1").split(",")
    if ip.strip()
)


def _rate_limit_key() -> str:
    remote_addr = request.remote_addr or "unknown"
    if remote_addr in TRUSTED_PROXIES:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[-1].strip()

    provided = request.headers.get("X-API-Key") or request.headers.get("Authorization")
    if provided:
        token = provided.split()[-1]
        return f"api:{token[:16]}"

    return remote_addr


_LIMITER_STORAGE_URI = (
    os.getenv("REX_SPEAK_STORAGE_URI")
    or os.getenv("FLASK_LIMITER_STORAGE_URI")
    or "memory://"
)

limiter = Limiter(
    key_func=_rate_limit_key,
    storage_uri=_LIMITER_STORAGE_URI,
    app=app,
    default_limits=[],
)

_RATE_CACHE: dict[str, deque] | None = defaultdict(deque) if _LIMITER_STORAGE_URI.startswith("memory") else None

app.register_blueprint(create_ha_blueprint())

# ------------------------------------------------------------------------------
# Config and Globals
# ------------------------------------------------------------------------------

USERS_MAP = load_users_map()
USER_PROFILES = load_all_profiles()
DEFAULT_USER = resolve_user_key(
    os.getenv("REX_ACTIVE_USER"), USERS_MAP, profiles=USER_PROFILES
) or (sorted(USER_PROFILES.keys())[0] if USER_PROFILES else "james")

USER_VOICES = {
    user: extract_voice_reference(profile, user_key=user)
    for user, profile in USER_PROFILES.items()
}
if DEFAULT_USER not in USER_VOICES:
    USER_VOICES[DEFAULT_USER] = None

_TTS_ENGINE: TTS | None = None

DEFAULT_TTS_MODEL = os.getenv("REX_TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
_MODEL_PATTERN = re.compile(r"^[\w\-./]+(\/[\w\-./]+)*$")
if not _MODEL_PATTERN.match(DEFAULT_TTS_MODEL):
    logger.warning("Invalid TTS model name '%s'; using default.", DEFAULT_TTS_MODEL)
    DEFAULT_TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

API_KEY: str | None = None
RATE_LIMIT = _parse_int("REX_SPEAK_RATE_LIMIT", os.getenv("REX_SPEAK_RATE_LIMIT"), default=30)
RATE_LIMIT_WINDOW = _parse_int("REX_SPEAK_RATE_WINDOW", os.getenv("REX_SPEAK_RATE_WINDOW"), default=60)
MAX_TEXT_LENGTH = _parse_int("REX_SPEAK_MAX_CHARS", os.getenv("REX_SPEAK_MAX_CHARS"), default=800)

if RATE_LIMIT > 0 and RATE_LIMIT_WINDOW > 0:
    _UNIT_MAP = {1: "second", 60: "minute", 3600: "hour", 86400: "day"}
    unit = _UNIT_MAP.get(RATE_LIMIT_WINDOW, None)
    if unit:
        _RATE_LIMIT_SPEC = f"{RATE_LIMIT} per {unit}"
    else:
        _RATE_LIMIT_SPEC = f"{RATE_LIMIT}/{RATE_LIMIT_WINDOW} second"
else:
    _RATE_LIMIT_SPEC = None

# ------------------------------------------------------------------------------
# Error Handlers
# ------------------------------------------------------------------------------


@app.errorhandler(AuthenticationError)
def _handle_auth_error(exc: AuthenticationError) -> Tuple[Response, int]:
    return jsonify({"error": str(exc)}), 401


@app.errorhandler(TextToSpeechError)
def _handle_tts_error(exc: TextToSpeechError) -> Tuple[Response, int]:
    return jsonify({"error": str(exc)}), 500


@app.errorhandler(RateLimitExceeded)
def _handle_rate_limit(exc: RateLimitExceeded) -> Response:
    response = jsonify({"error": "Too many requests"})
    response.status_code = 429
    if getattr(exc, "retry_after", None):
        response.headers["Retry-After"] = str(int(exc.retry_after))
    return response


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def _get_tts_engine() -> TTS:
    global _TTS_ENGINE
    if _TTS_ENGINE is None:
        _TTS_ENGINE = TTS(model_name=DEFAULT_TTS_MODEL, progress_bar=False)
    return _TTS_ENGINE


def _request_api_key() -> Optional[str]:
    provided = request.headers.get("X-API-Key") or request.headers.get("Authorization")
    if not provided:
        return None
    parts = provided.split()
    return parts[-1] if parts else None


def get_api_key() -> Optional[str]:
    return os.getenv("REX_SPEAK_API_KEY") or None


def _require_api_key() -> None:
    provided = _request_api_key()
    api_key = get_api_key()
    if not api_key:
        raise AuthenticationError("Missing API key")
    if not provided:
        raise AuthenticationError("Missing API key")
    if not hmac_compare(provided, api_key):
        raise AuthenticationError("Invalid API key")


def hmac_compare(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a.encode("utf-8"), b.encode("utf-8")):
        result |= x ^ y
    return result == 0


def _enforce_rate_limit() -> None:
    if RATE_LIMIT <= 0 or RATE_LIMIT_WINDOW <= 0:
        return
    if _RATE_CACHE is None:
        return
    now = time.monotonic()
    key = _rate_limit_key()
    bucket = _RATE_CACHE.setdefault(key, deque())
    while bucket and now - bucket[0] > RATE_LIMIT_WINDOW:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT:
        raise RateLimitExceeded("Too many requests")
    bucket.append(now)


def _resolve_speaker_wav(user_key: str) -> Optional[str]:
    voice_path = USER_VOICES.get(user_key)
    if voice_path and Path(voice_path).is_file():
        return voice_path
    return None


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------


@app.route("/speak", methods=["POST"])
def speak() -> Response:
    _require_api_key()
    _enforce_rate_limit()

    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not text or not isinstance(text, str):
        return jsonify({"error": "Text is required"}), 400
    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({"error": "Text exceeds maximum length"}), 400

    language = payload.get("language") or "en"
    user_key = payload.get("user") or DEFAULT_USER
    speaker_wav = _resolve_speaker_wav(user_key)

    engine = _get_tts_engine()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_path = tmp.name

    try:
        engine.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=output_path,
        )
    except Exception as exc:
        raise TextToSpeechError(str(exc)) from exc
    try:
        with open(output_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
    except Exception as exc:
        raise TextToSpeechError(str(exc)) from exc
    finally:
        try:
            Path(output_path).unlink(missing_ok=True)
        except PermissionError:
            logger.debug("Skipping cleanup for locked file: %s", output_path)

    return Response(audio_bytes, mimetype="audio/wav")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------


def main() -> None:
    if not get_api_key():
        raise RuntimeError("REX_SPEAK_API_KEY must be set")
    app.run(host="0.0.0.0", port=int(os.getenv("REX_SPEAK_PORT", "5005")))


if __name__ == "__main__":
    main()
