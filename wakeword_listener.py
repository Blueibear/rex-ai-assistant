"""Standalone wake-word listener used for quick manual testing."""

from __future__ import annotations

import os

import numpy as np
import simpleaudio as sa
import sounddevice as sd

from rex.wakeword_utils import detect_wakeword, load_wakeword_model
from rex.wake_acknowledgment import ensure_wake_acknowledgment_sound

# Configuration
WAKEWORD = os.getenv("REX_WAKEWORD", "rex")
THRESHOLD = float(os.getenv("REX_WAKEWORD_THRESHOLD", "0.5"))

# Load wake-word detection model
wake_model, wake_keyword = load_wakeword_model(keyword=WAKEWORD)

# Audio settings
sample_rate = 16000
block_size = sample_rate  # 1-second blocks

# Wake confirmation sound
wake_sound_path = ensure_wake_acknowledgment_sound()

print(f"🔊 Listening for wake word: '{wake_keyword}'…")


def play_confirmation_sound() -> None:
    try:
        wave_obj = sa.WaveObject.from_wave_file(wake_sound_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as exc:
        print(f"[!] Could not play wake confirmation sound: {exc}")


def listen_for_wakeword() -> bool:
    def audio_callback(indata, frames, time, status):
        if status:
            print("[!] Audio stream status:", status)
        audio_data = np.squeeze(indata)
        if detect_wakeword(wake_model, audio_data, threshold=THRESHOLD):
            print(f"✔ Wakeword detected: '{wake_keyword}'")
            play_confirmation_sound()
            raise StopIteration  # Stop the stream after detection

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
    except Exception as exc:
        print("[!] Wakeword listener error:", exc)
        return False


if __name__ == "__main__":
    success = listen_for_wakeword()
    if success:
        print("✅ Wakeword test completed.")
    else:
        print("❌ Wakeword test failed or aborted.")
