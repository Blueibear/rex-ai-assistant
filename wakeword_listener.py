"""Standalone wake-word listener used for quick manual testing."""

from __future__ import annotations

# Load .env before accessing any environment variables
from utils.env_loader import load as _load_env

_load_env()

import os

import numpy as np
import simpleaudio as sa
import sounddevice as sd

from rex.config import _parse_float
from rex.wake_acknowledgment import ensure_wake_acknowledgment_sound
from rex.wakeword_utils import detect_wakeword, load_wakeword_model

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

WAKEWORD = os.getenv("REX_WAKEWORD", "rex")
THRESHOLD = _parse_float("REX_WAKEWORD_THRESHOLD", os.getenv("REX_WAKEWORD_THRESHOLD"), default=0.5)
SAMPLE_RATE = 16000
BLOCK_SIZE = SAMPLE_RATE  # 1-second blocks

# ------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------

# Load model
wake_model, wake_keyword = load_wakeword_model(keyword=WAKEWORD)

# Load acknowledgment sound
wake_sound_path = ensure_wake_acknowledgment_sound()

print(f"🔊 Listening for wake word: '{wake_keyword}'…")


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def play_confirmation_sound() -> None:
    try:
        wave_obj = sa.WaveObject.from_wave_file(wake_sound_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as exc:
        print(f"[!] Could not play wake confirmation sound: {exc}")


def listen_for_wakeword() -> bool:
    """Start streaming from microphone and detect the wakeword in real time."""

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
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            callback=audio_callback,
        ):
            while True:
                sd.sleep(100)  # Sleep in small chunks
    except StopIteration:
        return True
    except Exception as exc:
        print("[!] Wakeword listener error:", exc)
        return False


# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    success = listen_for_wakeword()
    if success:
        print("✅ Wakeword test completed.")
    else:
        print("❌ Wakeword test failed or aborted.")

