import time
import os
import sounddevice as sd
import numpy as np
import simpleaudio as sa
from openwakeword.model import Model

# Load OpenWakeWord model (if not already loaded)
model = Model(backend="onnx")
wakeword = "rex"  # Use "rex" or any other wake word
model.load_wakeword(wakeword)

# Audio settings
sample_rate = 16000
duration = 1  # second
block_size = int(sample_rate * duration)

# Wake confirmation sound path
wake_sound_path = os.path.join(os.path.dirname(__file__), "assets", "rex_wake_acknowledgment (1).wav")

# Play the confirmation sound when wakeword is detected
def play_confirmation_sound():
    try:
        wave_obj = sa.WaveObject.from_wave_file(wake_sound_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"[!] Could not play wake confirmation sound: {e}")

# Callback function for listening to the audio stream
def audio_callback(indata, frames, time, status):
    if status:
        print("[!] Audio stream status:", status)
    audio_data = np.squeeze(indata)

    score = model.score(audio_data)
    if score > 0.5:
        print("👂 Wake word detected: Rex")
        play_confirmation_sound()

# Start listening for the wake word
with sd.InputStream(channels=1, samplerate=sample_rate, blocksize=block_size, callback=audio_callback):
    input("Press Enter to stop listening...\n")

