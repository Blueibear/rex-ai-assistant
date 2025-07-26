from flask import Flask, request, send_file, jsonify
from TTS.api import TTS
import os
import uuid

app = Flask(__name__)

# Define paths for user voice references. 
# Use None values when a specific speaker reference is not available.
USER_VOICES = {
    "james": None,
    "cole": None
}

# Load XTTS model at app startup
xtts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)

@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    text = data.get("text")
    user = data.get("user", "james").lower()

    if not text:
        return jsonify({"error": "Missing text parameter"}), 400

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
