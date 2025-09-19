"""Entry point for the Rex voice assistant demo.

This script provides a simple, self-contained example of how to tie together wake-word detection, speech-to-text
(STT) using OpenAI Whisper, transformer-driven responses via Hugging Face models, and text-to-speech (TTS) using
Coqui XTTS. It demonstrates a basic loop where a wake word triggers listening for a short command, transcribes the
user's utterance, generates a reply, speaks the reply aloud, and then continues listening.

To keep the example portable, all file paths are relative to the repository root. The wake confirmation sound comes
from the ``assets`` directory included with the project. If you wish to use custom speaker voices for the TTS module,
you can point the ``SPEAKER_VOICES`` dictionary at your own ``.wav`` files; otherwise Coqui XTTS will fall back to
its default speaker voice.

Usage::

    # install dependencies
    pip install -r requirements.txt

    # run the assistant
    python rex_assistant.py

Press ``Enter`` in the console to exit the program.
"""


import importlib
import importlib.util
import os
import tempfile
import textwrap
from typing import Optional

import numpy as np
import sounddevice as sd
import simpleaudio as sa
import soundfile as sf
import whisper
from TTS.api import TTS

from llm_client import LanguageModel
from wakeword_utils import detect_wakeword, load_wakeword_model

_WEB_SEARCH_SPEC = importlib.util.find_spec("plugins.web_search")
if _WEB_SEARCH_SPEC is not None:
    search_web = getattr(importlib.import_module("plugins.web_search"), "search_web", None)
else:
    search_web = None

from memory_utils import (
    extract_voice_reference,
    load_all_profiles,
    load_users_map,
    resolve_user_key,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

USERS_MAP = load_users_map()
USER_PROFILES = load_all_profiles()
ACTIVE_USER = resolve_user_key(os.getenv("REX_ACTIVE_USER"), USERS_MAP, profiles=USER_PROFILES)

if not ACTIVE_USER:
    if USER_PROFILES:
        ACTIVE_USER = sorted(USER_PROFILES.keys())[0]
    else:
        ACTIVE_USER = "james"

ACTIVE_PROFILE = USER_PROFILES.get(ACTIVE_USER, {})
ACTIVE_USER_DISPLAY = (
    ACTIVE_PROFILE.get("name") if isinstance(ACTIVE_PROFILE, dict) else None
)

# Wake word to listen for.  ``wakeword_utils`` falls back to bundled
# openWakeWord models when a custom ``rex.onnx`` is not present.
WAKEWORD = os.getenv("REX_WAKEWORD", "rex")
WAKEWORD_THRESHOLD = float(os.getenv("REX_WAKEWORD_THRESHOLD", "0.5"))

# Relative path to the wake confirmation sound.  This file is provided in
# the repository’s ``assets`` directory.  If you place the script
# elsewhere, adjust this path accordingly.
WAKE_SOUND_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "rex_wake_acknowledgment (1).wav"
)

# Mapping of user names to speaker reference WAV files.  Entries are loaded
# from the ``Memory`` directory so setting ``REX_ACTIVE_USER`` automatically
# selects the correct profile when the assistant starts.
SPEAKER_VOICES = {
    user: extract_voice_reference(profile)
    for user, profile in USER_PROFILES.items()
}

if ACTIVE_USER not in SPEAKER_VOICES:
    SPEAKER_VOICES[ACTIVE_USER] = None

# Duration (in seconds) to record the user’s utterance after the wake word
# triggers.  A real assistant could implement voice activity detection
# instead of a fixed duration.
COMMAND_DURATION = 5

# Whisper model size.  Valid options include "tiny", "base", "small",
# "medium" and "large".  Larger models yield higher accuracy at the
# expense of increased memory and compute requirements.
WHISPER_MODEL_NAME = os.getenv("REX_WHISPER_MODEL", "base")

LLM = LanguageModel()
ASSISTANT_PERSONA = textwrap.dedent(
    """
    You are Rex, a focused AI voice assistant that keeps responses concise.
    Reference the active user's preferences when it helps personalise your
    answer. Always respond in natural English prose.
    """
)



# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _build_profile_context() -> str:
    lines = []
    if isinstance(ACTIVE_PROFILE, dict):
        name = ACTIVE_PROFILE.get("name")
        if isinstance(name, str):
            lines.append(f"The active user is {name}.")
        role = ACTIVE_PROFILE.get("role")
        if isinstance(role, str):
            lines.append(f"Their role: {role}.")
        preferences = ACTIVE_PROFILE.get("preferences")
        if isinstance(preferences, dict):
            tone = preferences.get("tone")
            if isinstance(tone, str):
                lines.append(f"Preferred tone: {tone}.")
            topics = preferences.get("topics")
            if isinstance(topics, list):
                cleaned = [topic for topic in topics if isinstance(topic, str)]
                if cleaned:
                    lines.append(
                        "They are interested in: " + ", ".join(sorted(set(cleaned))) + "."
                    )
    return "\n".join(lines)


PROFILE_CONTEXT = _build_profile_context()


def _build_prompt(user_text: str) -> str:
    context = PROFILE_CONTEXT
    if context:
        return (
            f"{ASSISTANT_PERSONA}\n\n{context}\n\nUser: {user_text}\nAssistant:"
        )
    return f"{ASSISTANT_PERSONA}\n\nUser: {user_text}\nAssistant:"


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


def play_sound(path: str) -> None:
    """Play a WAV file synchronously."""

    try:
        wave_obj = sa.WaveObject.from_wave_file(path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as exc:
        print(f"[audio] Could not play sound '{path}': {exc}")


def generate_response(text: str) -> str:
    """Generate a natural-language reply using the configured language model."""

    prompt = _build_prompt(text)
    try:
        completion = LLM.generate(prompt)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[llm] Error generating response: {exc}")
        return f"I heard: {text}. Could you repeat that while I restart my thoughts?"
    return completion


def handle_command(text: str) -> str:
    """Handle a transcribed command and return a response.

    If the command starts with the word "search" and a web search
    function is available, the assistant performs a search and returns
    the first result.  Otherwise it simply echoes the user’s input via
    ``generate_response``.

    Parameters
    ----------
    text: str
        The user’s transcribed utterance.

    Returns
    -------
    str
        The reply to speak.
    """
    lower = text.strip().lower()
    if lower.startswith("search ") and search_web is not None:
        query = text[7:].strip()
        result = search_web(query)
        return result or "No result found."
    return generate_response(text)


def speak(text: str, user: Optional[str] = None) -> None:
    """Convert text to speech and play it.

    This uses Coqui’s XTTS model.  If a speaker reference WAV exists
    in ``SPEAKER_VOICES`` for the given user, it will be used to
    condition the voice; otherwise the model’s default speaker is
    used.

    Parameters
    ----------
    text: str
        Text to convert to speech.
    user: str, optional
        Name of the user whose voice sample should be used.  Defaults
        to the profile selected via ``REX_ACTIVE_USER``.
    """
    target_user = (user or ACTIVE_USER).lower()
    if target_user not in SPEAKER_VOICES:
        target_user = ACTIVE_USER

    speaker_wav: Optional[str] = SPEAKER_VOICES.get(target_user)
    if speaker_wav and not os.path.isfile(speaker_wav):
        speaker_wav = None
    tts = _get_tts_engine()
    # Create a temporary file for the audio output
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_path = tmp.name
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language="en",
        file_path=output_path,
    )
    play_sound(output_path)
    try:
        os.remove(output_path)
    except OSError:
        pass


def main() -> None:
    """Main event loop for the assistant."""
    if ACTIVE_USER_DISPLAY:
        print(f"[config] Active user: {ACTIVE_USER_DISPLAY} ({ACTIVE_USER})")
    else:
        print(f"[config] Active user: {ACTIVE_USER}")

    # Load wake-word model
    wake_model, wake_keyword = load_wakeword_model(keyword=WAKEWORD)
    print(f"[setup] Wake word active: {wake_keyword}")

    # Load the Whisper STT model
    print(f"[setup] Loading Whisper model '{WHISPER_MODEL_NAME}'…")
    stt_model = whisper.load_model(WHISPER_MODEL_NAME)
    print("[setup] Whisper loaded.")

    sample_rate = 16000
    block_size = sample_rate  # 1 second blocks

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        # Flatten multi-channel audio
        audio_data = np.squeeze(indata)
        if detect_wakeword(wake_model, audio_data, threshold=WAKEWORD_THRESHOLD):
            print("[wake] Wake word detected.")
            play_sound(WAKE_SOUND_PATH)

            # Record the user’s command
            print(f"[listen] Recording for {COMMAND_DURATION} seconds…")
            recording = sd.rec(int(COMMAND_DURATION * sample_rate),
                               samplerate=sample_rate,
                               channels=1)
            sd.wait()
            command_audio = recording[:, 0]
            # Write to temp file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, command_audio, sample_rate)
                audio_path = tmp.name
            # Transcribe
            result = stt_model.transcribe(audio_path)
            os.remove(audio_path)
            text = result.get("text", "").strip()
            if not text:
                print("[stt] No speech detected.")
                return
            print(f"[stt] You said: {text}")
            # Handle the command
            reply = handle_command(text)
            print(f"[reply] {reply}")
            # Speak the reply
            speak(reply)

    # Start streaming microphone audio
    print("[info] Rex Assistant is listening.  Say the wake word to begin.")
    with sd.InputStream(
        channels=1,
        samplerate=sample_rate,
        blocksize=block_size,
        callback=audio_callback,
    ):
        input("Press Enter to stop listening…\n")


if __name__ == "__main__":
    main()
