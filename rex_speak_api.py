import hmac
import logging
import os
import re
import tempfile
import time
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple

from flask import Flask, request, send_file, jsonify, after_this_request
from TTS.api import TTS

from memory_utils import (
    extract_voice_reference,
    load_all_profiles,
    load_users_map,
    resolve_user_key,
)

app = Flask(__name__)


logger = logging.getLogger("rex.speak_api")

if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("REX_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


USERS_MAP = load_users_map()
USER_PROFILES = load_all_profiles()
DEFAULT_USER = resolve_user_key(
    os.getenv("REX_ACTIVE_USER"), USERS_MAP, profiles=USER_PROFILES
)

if not DEFAULT_USER:
    if USER_PROFILES:
        DEFAULT_USER = sorted(USER_PROFILES.keys())[0]
    else:
        DEFAULT_USER = "james"

USER_VOICES = {
    user: extract_voice_reference(profile, user_key=user)
    for user, profile in USER_PROFILES.items()
}

if DEFAULT_USER not in USER_VOICES:
    USER_VOICES[DEFAULT_USER] = None

_TTS_ENGINE: Optional[TTS] = None


_MODEL_PATTERN = re.compile(r"^[\w\-./]+(\/[\w\-./]+)*$")
DEFAULT_TTS_MODEL = os.getenv(
    "REX_TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"
)
if not _MODEL_PATTERN.match(DEFAULT_TTS_MODEL):
    logger.warning(
        "Invalid TTS model name '%s'; falling back to default.", DEFAULT_TTS_MODEL
    )
    DEFAULT_TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

API_KEY = os.getenv("REX_SPEAK_API_KEY")
RATE_LIMIT = int(os.getenv("REX_SPEAK_RATE_LIMIT", "30"))
RATE_LIMIT_WINDOW = int(os.getenv("REX_SPEAK_RATE_WINDOW", "60"))
MAX_TEXT_LENGTH = int(os.getenv("REX_SPEAK_MAX_CHARS", "800"))

_RATE_STATE: Dict[str, deque] = defaultdict(deque)


def _prune_requests(identity: str, now: float) -> deque:
    window = RATE_LIMIT_WINDOW
    entries = _RATE_STATE[identity]
    while entries and now - entries[0] > window:
        entries.popleft()
    return entries


def _check_rate_limit(identity: str) -> Tuple[bool, int]:
    if RATE_LIMIT <= 0:
        return True, 0
    now = time.monotonic()
    entries = _prune_requests(identity, now)
    if len(entries) >= RATE_LIMIT:
        retry = int(max(0, RATE_LIMIT_WINDOW - (now - entries[0])))
        return False, retry
    entries.append(now)
    return True, 0


def _get_request_identity(provided_key: Optional[str]) -> str:
    if provided_key:
        return f"key:{provided_key[:8]}"
    remote_addr = request.headers.get("X-Forwarded-For") or request.remote_addr
    return remote_addr or "anonymous"


def _extract_api_key(payload: dict | None) -> Optional[str]:
    header_key = request.headers.get("X-API-Key")
    if header_key:
        return header_key
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:]
    if payload and "api_key" in payload:
        return str(payload["api_key"])
    query_key = request.args.get("api_key")
    if query_key:
        return query_key
    return None


def _require_api_key(provided_key: Optional[str]) -> bool:
    if not API_KEY:
        return True
    if not provided_key:
        return False
    try:
        return hmac.compare_digest(provided_key, API_KEY)
    except TypeError:
        return False


def _validate_text(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return "Text must be a string."
    normalised = text.strip()
    if not normalised:
        return "Text must not be empty."
    if len(normalised) > MAX_TEXT_LENGTH:
        return f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters."
    return None


def _select_speaker(user: Optional[str]) -> Optional[str]:
    if user:
        candidate = str(user).lower()
    else:
        candidate = DEFAULT_USER
    if candidate not in USER_VOICES:
        candidate = DEFAULT_USER
    speaker_wav = USER_VOICES.get(candidate)
    if speaker_wav and not os.path.isfile(speaker_wav):
        logger.warning("Speaker reference '%s' is missing; falling back to default.", speaker_wav)
        return None
    return speaker_wav


def _get_tts_engine() -> TTS:
    global _TTS_ENGINE
    if _TTS_ENGINE is None:
        logger.info("Loading TTS model '%s'", DEFAULT_TTS_MODEL)
        _TTS_ENGINE = TTS(
            model_name=DEFAULT_TTS_MODEL,
            progress_bar=False,
            gpu=False,
        )
    return _TTS_ENGINE


@app.route("/speak", methods=["POST"])
def speak():
    payload = request.get_json(silent=True) or {}
    provided_key = _extract_api_key(payload)
    if not _require_api_key(provided_key):
        logger.warning("Rejected request without valid API key.")
        return jsonify({"error": "Unauthorized"}), 401

    identity = _get_request_identity(provided_key)
    allowed, retry_after = _check_rate_limit(identity)
    if not allowed:
        logger.warning("Rate limit exceeded for %s", identity)
        response = jsonify({"error": "Too many requests"})
        response.status_code = 429
        response.headers["Retry-After"] = str(retry_after)
        return response

    text = payload.get("text")
    validation_error = _validate_text(text)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    user_param = payload.get("user")
    speaker_wav = _select_speaker(user_param)

    language = payload.get("language", "en")
    if not isinstance(language, str):
        return jsonify({"error": "language must be a string"}), 400

    logger.info("Generating speech for %s (language=%s)", identity, language)

    tts_engine = _get_tts_engine()

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_path = temp_file.name
    temp_file.close()

    try:
        tts_engine.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=output_path,
        )

        @after_this_request
        def cleanup(response):  # pragma: no cover - exercised in integration
            try:
                os.remove(output_path)
            except OSError:
                logger.debug("Could not remove temporary file '%s'", output_path)
            return response

        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="rex_response.wav",
        )
    except Exception as exc:
        logger.exception("TTS generation failed")
        if os.path.exists(output_path):
            os.remove(output_path)
        return jsonify({"error": str(exc)}), 500
