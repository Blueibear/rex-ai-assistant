import os
os.environ["OPENWAKEWORD_BACKEND"] = "onnx"

import sounddevice as sd
import numpy as np
import simpleaudio as sa
from openwakeword.model import Model

# 🛑 Don't let it try to use TFLite at all
model = Model(backend="onnx", enable_tflite=False) # Only ONNX
wakeword_model_path = "rex.onnx"
model.load_wakeword(wakeword_model_path)

# Audio settings
sample_rate = 16000
duration = 1  # seconds per chunk
block_size = int(sample_rate * duration)

# Wake confirmation sound path
wake_sound_path = "rex_wake_confirmation.wav"

print("🎙️ Listening for wake word: 'Rex'...")

def play_confirmation_sound():
    try:
        wave_obj = sa.WaveObject.from_wave_file(wake_sound_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"[!] Could not play wake confirmation sound: {e}")

def listen_for_wakeword():
    def audio_callback(indata, frames, time, status):
        if status:
            print("[!] Audio stream status:", status)
        audio_data = np.squeeze(indata)
        score = model.score(audio_data)
        if score > 0.5:
            print("👂 Wake word detected: Rex")
            play_confirmation_sound()
            raise StopIteration  # Exit audio stream when detected

    try:
        with sd.InputStream(
            channels=1,
            samplerate=sample_rate,
            blocksize=block_size,
            callback=audio_callback,
        ):
            while True:
                sd.sleep(100)
    except StopIteration:
        return True
    except Exception as e:
        print("[!] Wakeword listener error:", e)
        return False





