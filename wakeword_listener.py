"""Standalone wake-word listener used for quick manual testing."""

from __future__ import annotations

import os

import numpy as np
import simpleaudio as sa
import sounddevice as sd

from config import load_config
from wakeword_utils import detect_wakeword, load_wakeword_model

CONFIG = load_config()
WAKEWORD = os.getenv("REX_WAKEWORD", CONFIG.wakeword)
THRESHOLD = float(os.getenv("REX_WAKEWORD_THRESHOLD", str(CONFIG.wakeword_threshold)))

wake_model, wake_keyword = load_wakeword_model(
    keyword=WAKEWORD,
    sensitivity=CONFIG.wakeword_sensitivity,
)

# Audio settings aligned with Porcupine expectations
sample_rate = wake_model.sample_rate
block_size = wake_model.frame_length * 8

# Wake confirmation sound path (use relative path to the assets folder)
wake_sound_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "rex_wake_acknowledgment (1).wav",
)

print(f"Listening for wake word: '{wake_keyword}'")


def play_confirmation_sound() -> None:
    try:
        wave_obj = sa.WaveObject.from_wave_file(wake_sound_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as exc:  # pragma: no cover - best effort logging
        print(f"[!] Could not play wake confirmation sound: {exc}")


def listen_for_wakeword() -> bool:
    def audio_callback(indata, frames, time, status):
        if status:
            print("[!] Audio stream status:", status)
        audio_data = np.squeeze(indata)
        if detect_wakeword(wake_model, audio_data, threshold=THRESHOLD):
            print(f"Wakeword detected: '{wake_keyword}'")
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
    except Exception as exc:  # pragma: no cover - hardware specific
        print("[!] Wakeword listener error:", exc)
        return False


if __name__ == "__main__":
    listen_for_wakeword()
