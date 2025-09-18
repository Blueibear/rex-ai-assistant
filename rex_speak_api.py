import os
import uuid

from flask import Flask, request, send_file, jsonify
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
    user: extract_voice_reference(profile)
    for user, profile in USER_PROFILES.items()
}

if DEFAULT_USER not in USER_VOICES:
    USER_VOICES[DEFAULT_USER] = None

# Load XTTS model at app startup
xtts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)


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

    # If a speaker file path is provided but the file doesn't exist, return an error
    if speaker_wav and not os.path.exists(speaker_wav):
        return jsonify({"error": f"Speaker reference file not found for user '{user}'"}), 404

    # Generate a unique filename in the current directory for the response audio
    output_filename = f"rex_response_{uuid.uuid4().hex}.wav"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)

    try:
        # Generate speech to the file
        xtts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language="en",
            file_path=output_path,
        )

        return send_file(output_path, mimetype="audio/wav", as_attachment=True, download_name="rex_response.wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Remove the temporary file
        if os.path.exists(output_path):
            os.remove(output_path)
