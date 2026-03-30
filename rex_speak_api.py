"""Hardened Flask API for text-to-speech synthesis."""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
import time
from collections import defaultdict, deque
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

from flask import Flask, Response, request
from flask_cors import CORS

from rex.config import _parse_int, load_config
from rex.exception_handler import wrap_entrypoint
from rex.graceful_shutdown import get_shutdown_handler
from rex.ha_bridge import create_blueprint as create_ha_blueprint
from rex.health import check_config, create_health_blueprint
from rex.http_errors import (
    BAD_REQUEST,
    INTERNAL_ERROR,
    PAYLOAD_TOO_LARGE,
    TOO_MANY_REQUESTS,
    UNAUTHORIZED,
    error_response,
)
from rex.request_logging import install_request_logging
from rex.startup_validation import check_startup_env
from rex.tts_utils import chunk_text_for_xtts

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
install_request_logging(app)
app.register_blueprint(create_health_blueprint(checks=[check_config]))

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

TRUSTED_PROXIES = {
    ip.strip() for ip in os.getenv("REX_TRUSTED_PROXIES", "127.0.0.1,::1").split(",") if ip.strip()
}


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
    os.getenv("REX_SPEAK_STORAGE_URI") or os.getenv("FLASK_LIMITER_STORAGE_URI") or "memory://"
)

limiter = Limiter(
    key_func=_rate_limit_key,
    storage_uri=_LIMITER_STORAGE_URI,
    app=app,
    default_limits=[],
)

_RATE_CACHE: dict[str, deque] | None = (
    defaultdict(deque) if _LIMITER_STORAGE_URI.startswith("memory") else None
)

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
    user: extract_voice_reference(profile, user_key=user) for user, profile in USER_PROFILES.items()
}
if DEFAULT_USER not in USER_VOICES:
    USER_VOICES[DEFAULT_USER] = None

_TTS_ENGINE: TTS | None = None
_tts_lock = threading.Lock()

DEFAULT_TTS_MODEL = os.getenv("REX_TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
_MODEL_PATTERN = re.compile(r"^[\w\-./]+(\/[\w\-./]+)*$")
if not _MODEL_PATTERN.match(DEFAULT_TTS_MODEL):
    logger.warning("Invalid TTS model name '%s'; using default.", DEFAULT_TTS_MODEL)
    DEFAULT_TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

API_KEY: str | None = None
RATE_LIMIT = _parse_int("REX_SPEAK_RATE_LIMIT", os.getenv("REX_SPEAK_RATE_LIMIT"), default=30)
RATE_LIMIT_WINDOW = _parse_int(
    "REX_SPEAK_RATE_WINDOW", os.getenv("REX_SPEAK_RATE_WINDOW"), default=60
)
MAX_TEXT_LENGTH = _parse_int("REX_SPEAK_MAX_CHARS", os.getenv("REX_SPEAK_MAX_CHARS"), default=800)
MAX_REQUEST_BYTES = _parse_int(
    "REX_SPEAK_MAX_REQUEST_BYTES",
    os.getenv("REX_SPEAK_MAX_REQUEST_BYTES"),
    default=65536,  # 64 KB
)
TTS_SPEED = load_config().tts_speed

# Enforce body-size limit via Flask: rejects Content-Length violations before the body
# is read and caps streamed reads at MAX_REQUEST_BYTES for requests without Content-Length.
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES

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
# Graceful-shutdown guard
# ------------------------------------------------------------------------------


@app.before_request
def _reject_during_shutdown() -> tuple[Response, int] | None:
    """Return 503 when a SIGTERM-triggered shutdown is in progress."""
    if get_shutdown_handler().is_shutting_down:
        resp, status = error_response("SERVICE_UNAVAILABLE", "Server is shutting down", 503)
        return resp, status
    return None


@app.before_request
def _check_request_size() -> tuple[Response, int] | None:
    """Reject requests whose Content-Length exceeds MAX_REQUEST_BYTES before reading body."""
    cl = request.content_length
    if cl is not None and cl > MAX_REQUEST_BYTES:
        resp, status = error_response(PAYLOAD_TOO_LARGE, "Request body too large", 413)
        return resp, status
    return None


# Error Handlers
# ------------------------------------------------------------------------------


@app.errorhandler(AuthenticationError)
def _handle_auth_error(exc: AuthenticationError) -> tuple[Response, int]:
    resp, status = error_response(UNAUTHORIZED, str(exc), 401)
    resp.status_code = status
    return resp  # type: ignore[return-value]


@app.errorhandler(TextToSpeechError)
def _handle_tts_error(exc: TextToSpeechError) -> tuple[Response, int]:
    resp, status = error_response(INTERNAL_ERROR, str(exc), 500)
    resp.status_code = status
    return resp  # type: ignore[return-value]


@app.errorhandler(RateLimitExceeded)
def _handle_rate_limit(exc: RateLimitExceeded) -> Response:
    resp, _ = error_response(TOO_MANY_REQUESTS, "Too many requests", 429)
    resp.status_code = 429
    if getattr(exc, "retry_after", None):
        resp.headers["Retry-After"] = str(int(exc.retry_after))
    return resp


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def _get_tts_engine() -> TTS:
    """Return the TTS engine singleton. Must be called while holding _tts_lock."""
    global _TTS_ENGINE
    if _TTS_ENGINE is None:
        if find_spec("TTS") is None:
            raise TextToSpeechError(
                'Coqui TTS is not installed. Install with `pip install -e ".[ml]"`.'
            )
        try:
            from rex.compat import ensure_transformers_compatibility

            ensure_transformers_compatibility()
            tts_class = import_module("TTS.api").TTS
        except Exception as exc:
            raise TextToSpeechError(
                'Failed to import Coqui TTS. Install with `pip install -e ".[ml]"`.'
            ) from exc
        _TTS_ENGINE = tts_class(model_name=DEFAULT_TTS_MODEL, progress_bar=False)
    return _TTS_ENGINE


def _load_audio_dependencies():
    if find_spec("numpy") is None or find_spec("soundfile") is None:
        raise TextToSpeechError(
            'Audio dependencies missing. Install with `pip install -e ".[audio]"`.'
        )
    try:
        np = import_module("numpy")
        sf = import_module("soundfile")
    except Exception as exc:
        raise TextToSpeechError(
            'Failed to import audio dependencies. Install with `pip install -e ".[audio]"`.'
        ) from exc
    return np, sf


def _request_api_key() -> str | None:
    provided = request.headers.get("X-API-Key") or request.headers.get("Authorization")
    if not provided:
        return None
    parts = provided.split()
    return parts[-1] if parts else None


def get_api_key() -> str | None:
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


def _resolve_speaker_wav(user_key: str) -> str | None:
    voice_path = USER_VOICES.get(user_key)
    if voice_path and Path(voice_path).is_file():
        return voice_path
    return None


def generate_speech(text: str, language: str, user_key: str) -> bytes:
    """Generate speech audio from text. Can be monkeypatched for testing."""
    speaker_wav = _resolve_speaker_wav(user_key)
    np, sf = _load_audio_dependencies()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_path = tmp.name

    # _tts_lock is acquired before initialization check and held until synthesis completes
    with _tts_lock:
        engine = _get_tts_engine()
        try:
            chunks = chunk_text_for_xtts(text, max_tokens=300)
            if not chunks:
                raise TextToSpeechError("No speech content to synthesize.")

            audio_segments = []
            sample_rate = None

            for chunk in chunks:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    chunk_path = tmp.name

                try:
                    engine.tts_to_file(
                        text=chunk,
                        speaker_wav=speaker_wav,
                        language=language,
                        file_path=chunk_path,
                        speed=TTS_SPEED,
                    )
                    data, rate = sf.read(chunk_path, dtype="float32")
                    if sample_rate is None:
                        sample_rate = rate
                    audio_segments.append(data)
                finally:
                    try:
                        Path(chunk_path).unlink(missing_ok=True)
                    except PermissionError:
                        logger.debug("Skipping cleanup for locked file: %s", chunk_path)

            if not audio_segments or sample_rate is None:
                raise TextToSpeechError("XTTS produced no audio output.")

            combined_audio = np.concatenate(audio_segments, axis=0)
            sf.write(output_path, combined_audio, sample_rate)
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

    return audio_bytes


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
        return error_response(BAD_REQUEST, "Text is required", 400)
    if len(text) > MAX_TEXT_LENGTH:
        return error_response(BAD_REQUEST, "Text exceeds maximum length", 400)

    language = payload.get("language") or "en"
    user_key = payload.get("user") or DEFAULT_USER

    audio_bytes = generate_speech(text, language, user_key)
    return Response(audio_bytes, mimetype="audio/wav")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------


@wrap_entrypoint
def main() -> None:
    check_startup_env()
    if not get_api_key():
        raise RuntimeError("REX_SPEAK_API_KEY must be set")
    shutdown = get_shutdown_handler()
    shutdown.install()
    app.run(host="127.0.0.1", port=int(os.getenv("REX_SPEAK_PORT") or "5005"))


if __name__ == "__main__":
    main()
if TYPE_CHECKING:  # pragma: no cover - typing only
    from TTS.api import TTS
