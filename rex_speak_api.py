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

try:
    from rex.memory_utils import (
        extract_voice_reference,
        load_all_profiles,
        load_users_map,
        resolve_user_key,
    )
except ImportError:  # pragma: no cover - fallback for module-level execution
    from memory_utils import (
        extract_voice_reference,
        load_all_profiles,
        load_users_map,
        resolve_user_key,
    )
from rex.assistant_errors import AuthenticationError, TextToSpeechError
from rex.config import settings

# ------------------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------------------

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logger = logging.getLogger("rex.speak_api")

if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("REX_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

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

_TTS_ENGINE: Optional[TTS] = None

DEFAULT_TTS_MODEL = os.getenv(
    "REX_TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"
)
_MODEL_PATTERN = re.compile(r"^[\w\-./]+(\/[\w\-./]+)*$")
if not _MODEL_PATTERN.match(DEFAULT_TTS_MODEL):
    logger.warning("Invalid TTS model name '%s'; using default.", DEFAULT_TTS_MODEL)
    DEFAULT_TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

API_KEY = os.getenv("REX_SPEAK_API_KEY")
RATE_LIMIT = int(os.getenv("REX_SPEAK_RATE_LIMIT", "30"))
RATE_LIMIT_WINDOW = int(os.getenv("REX_SPEAK_RATE_WINDOW", "60"))
MAX_TEXT_LENGTH = int(os.getenv("REX_SPEAK_MAX_CHARS", "800"))

if not API_KEY:
    raise RuntimeError("REX_SPEAK_API_KEY must be set before starting the speech API.")

if RATE_LIMIT > 0 and RATE_LIMIT_WINDOW > 0:
    _UNIT_MAP = {1: "second", 60: "minute", 3600: "hour", 86400: "day"}
    unit = _UNIT_MAP.get(RATE_LIMIT_WINDOW, None)
    if unit:
        _RATE_LIMIT_SPEC = f"{RATE_LIMIT} per {unit}"
    else:
        _RATE_LIMIT_SPEC = f"{RATE_LIMIT}/{RATE_LIMIT_WINDOW} second"
else:
    _RATE_LIMIT_SPEC = None

_LIMITER_STORAGE_URI = (
    os.getenv("REX_SPEAK_STORAGE_URI")
    or os.getenv("FLASK_LIMITER_STORAGE_URI")
    or "memory://"
)

limiter = Limiter(
    key_func=lambda: _rate_limit_key(),
    storage_uri=_LIMITER_STORAGE_URI,
    app=app,
    default_limits=[],
)

if _LIMITER_STORAGE_URI.startswith("memory"):
    _RATE_CACHE: dict[str, deque] = defaultdict(deque)
else:
    _RATE_CACHE = None

# ------------------------------------------------------------------------------
# Helpers
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
    retry_after = getattr(exc, "retry_after", None)
    if retry_after:
        response.headers["Retry-After"] = str(int(retry_after))
    return response


def _rate_limit_key() -> str:
    """Derive a stable rate-limit identity from API key or source address."""
    provided = request.headers.get("X-API-Key") or request.headers.get("Authorization")
    if provided:
        token = provided.split()[-1]
        return f"api:{token[:16]}"
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "anonymous"


def _extract_api_key(payload: dict | None) -> Optional[str]:
    """Extract API key from headers or payload."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return (
        request.headers.get("X-API-Key")
        or request.args.get("api_key")
        or (payload.get("api_key") if payload else None)
    )


def _require_api_key(provided_key: Optional[str]) -> bool:
    """Securely validate API key using constant-time comparison."""
    if not provided_key:
        return False
    try:
        return hmac.compare_digest(provided_key, API_KEY)
    except TypeError:
        return False


def _prune_requests(identity: str, now: float) -> deque:
    """Remove expired rate limit entries."""
    entries = _RATE_CACHE[identity]
    while entries and now - entries[0] > RATE_LIMIT_WINDOW:
        entries.popleft()
    return entries


def _check_rate_limit(identity: str) -> Tuple[bool, int]:
    """Check if request is within rate limit."""
    if _RATE_CACHE is None or RATE_LIMIT <= 0 or RATE_LIMIT_WINDOW <= 0:
        return True, 0
    now = time.monotonic()
    entries = _prune_requests(identity, now)
    if len(entries) >= RATE_LIMIT:
        retry = int(max(0, RATE_LIMIT_WINDOW - (now - entries[0])))
        return False, retry
    entries.append(now)
    return True, 0


def _validate_text(text: str) -> Optional[str]:
    """Validate text input."""
    if not isinstance(text, str):
        return "Text must be a string."
    normalised = text.strip()
    if not normalised:
        return "Text must not be empty."
    if len(normalised) > MAX_TEXT_LENGTH:
        return f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters."
    return None


def _sanitize_path(path: str) -> str:
    """Sanitize file path to prevent traversal attacks."""
    # Remove any parent directory references
    clean_path = os.path.normpath(path)
    if ".." in Path(clean_path).parts:
        raise ValueError("Path traversal detected")
    return clean_path


def _select_speaker(user: Optional[str]) -> Optional[str]:
    """Select speaker voice file for user."""
    candidate = str(user).lower() if user else DEFAULT_USER
    if candidate not in USER_VOICES:
        candidate = DEFAULT_USER
    speaker_wav = USER_VOICES.get(candidate)
    if speaker_wav:
        try:
            speaker_wav = _sanitize_path(speaker_wav)
            if not os.path.isfile(speaker_wav):
                logger.warning("Speaker reference '%s' is missing.", speaker_wav)
                return None
        except ValueError as exc:
            logger.error("Invalid speaker path: %s", exc)
            return None
    return speaker_wav


def _get_tts_engine() -> TTS:
    """Get or initialize TTS engine."""
    global _TTS_ENGINE
    if _TTS_ENGINE is None:
        logger.info("Loading TTS model '%s'", DEFAULT_TTS_MODEL)
        _TTS_ENGINE = TTS(
            model_name=DEFAULT_TTS_MODEL,
            progress_bar=False,
            gpu=False,
        )
    return _TTS_ENGINE


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------


def _apply_rate_limit(func):
    """Apply rate limiting to endpoint if configured."""
    if _RATE_LIMIT_SPEC:
        return limiter.limit(_RATE_LIMIT_SPEC)(func)
    return func


@app.route("/speak", methods=["POST"])
@_apply_rate_limit
def speak() -> Response:
    """Generate speech from text."""
    payload = request.get_json(silent=True) or {}
    provided_key = _extract_api_key(payload)

    if not _require_api_key(provided_key):
        logger.warning("Unauthorized request from %s", request.remote_addr)
        raise AuthenticationError("Missing or invalid API key")

    allowed, retry_after = _check_rate_limit(_rate_limit_key())
    if not allowed:
        response = jsonify({"error": "Too many requests"})
        response.status_code = 429
        if retry_after:
            response.headers["Retry-After"] = str(retry_after)
        return response

    text = payload.get("text")
    validation_error = _validate_text(text)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    speaker_wav = _select_speaker(payload.get("user"))
    language = payload.get("language", "en")

    logger.info("Generating speech (lang=%s, user=%s)", language, payload.get("user"))
    output_path = tempfile.mktemp(suffix=".wav")

    try:
        _get_tts_engine().tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=output_path,
        )

        @after_this_request
        def cleanup(response: Response) -> Response:
            try:
                os.remove(output_path)
            except OSError:
                logger.debug("Failed to remove temp file %s", output_path)
            return response

        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="rex_response.wav",
        )

    except Exception as exc:
        logger.exception("TTS generation failed.")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise TextToSpeechError(str(exc))


@app.route("/health", methods=["GET"])
def health() -> Response:
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "rex-speak-api"})


# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
