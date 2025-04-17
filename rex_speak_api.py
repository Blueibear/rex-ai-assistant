from flask import Flask, request, send_file, jsonify
from TTS.api import TTS
import os
import uuid

app = Flask(__name__)

# Define paths for user voice references
USER_VOICES = {
    "james": "C:/AI/CoquiXTTS/cleaned_jensen.wav",
    "cole": "C:/AI/CoquiXTTS/cole_voice.wav"
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
    if not speaker_wav or not os.path.exists(speaker_wav):
        return jsonify({"error": f"No speaker reference found for user '{user}'"}), 404

    output_filename = f"rex_response_{uuid.uuid4().hex}.wav"
    output_path = os.path.join("C:/AI/CoquiXTTS", output_filename)

    try:
        xtts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language="en",
            file_path=output_path
        )
        return send_file(output_path, mimetype="audio/wav", as_attachment=True, download_name="rex_response.wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)

if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0")
