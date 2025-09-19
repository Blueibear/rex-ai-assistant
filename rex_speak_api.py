import os
import tempfile
from typing import Optional

from flask import Flask, request, send_file, jsonify, after_this_request
from TTS.api import TTS

from memory_utils import (
    extract_voice_reference,
    load_all_profiles,
    load_users_map,
    resolve_user_key,
)

app = Flask(__name__)

USERS_MAP = load_users_map()
USER_PROFILES = load_all_profiles()
DEFAULT_USER = resolve_user_key(os.getenv("REX_ACTIVE_USER"), USERS_MAP, profiles=USER_PROFILES)

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


def _get_tts_engine() -> TTS:
    global _TTS_ENGINE
    if _TTS_ENGINE is None:
        _TTS_ENGINE = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False,
            gpu=False,
        )
    return _TTS_ENGINE


@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json() or {}
    text = data.get("text")
    user_param = data.get("user")

    if not text:
        return jsonify({"error": "Missing text parameter"}), 400

    if user_param:
        user = str(user_param).lower()
    else:
        user = DEFAULT_USER

    if user not in USER_VOICES:
        user = DEFAULT_USER

    speaker_wav = USER_VOICES.get(user)
    if speaker_wav and not os.path.isfile(speaker_wav):
        speaker_wav = None

    tts_engine = _get_tts_engine()

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_path = temp_file.name
    temp_file.close()

    try:
        tts_engine.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language="en",
            file_path=output_path,
        )

        @after_this_request
        def cleanup(response):  # pragma: no cover - exercised in integration
            try:
                os.remove(output_path)
            except OSError:
                pass
            return response

        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="rex_response.wav",
        )
    except Exception as exc:
        if os.path.exists(output_path):
            os.remove(output_path)
        return jsonify({"error": str(exc)}), 500
