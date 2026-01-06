"""Hardened Flask API for text-to-speech synthesis."""

from __future__ import annotations

import hmac
import logging
import os
import re
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional, Tuple

from flask import Flask, Response, jsonify, request, send_file, after_this_request
from flask_cors import CORS

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
    provided = request.headers.get("X-API-Key") or request.headers.get("Authorization")
    if provided:
        token = provided.split()[-1]
        return f"api:{token[:16]}"

    remote_addr = request.remote_addr or "unknown"
    if remote_addr in TRUSTED_PROXIES:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[-1].strip()

    return remote_addr

_LIMITER_STORAGE_URI = (
    os.getenv("REX_SPEAK_STORAGE_URI") or
    os.getenv("FLASK_LIMITER_STORAGE_URI") or
    "memory://"
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

API_KEY = os.getenv("REX_SPEAK_API_KEY")
RATE_LIMIT = int(os.getenv("REX_SPEAK_RATE_LIMIT", "30"))
RATE_LIMIT_WINDOW = int(os.getenv("REX_SPEAK_RATE_WINDOW", "60"))
MAX_TEXT_LENGTH = int(os.getenv("REX_SPEAK_MAX_CHARS", "800"))

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
