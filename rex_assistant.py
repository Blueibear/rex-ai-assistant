"""Voice assistant loop combining wake-word detection, Whisper STT, transformer replies, and XTTS TTS."""

import importlib
import importlib.util
import logging
import os
import tempfile
import textwrap
import threading
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import simpleaudio as sa
import soundfile as sf
import whisper
from TTS.api import TTS

from conversation_memory import ConversationMemory
from llm_client import LanguageModel
from wakeword_utils import detect_wakeword, load_wakeword_model
from wake_acknowledgment import ensure_wake_acknowledgment_sound

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


if not logging.getLogger("rex.assistant").handlers:
    logging.basicConfig(
        level=os.getenv("REX_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

logger = logging.getLogger("rex.assistant")


class FunctionRouter:
    """Route recognised commands to deterministic functions."""

    def __init__(self) -> None:
        self._routes: list[tuple[Callable[[str], bool], Callable[[str], str]]] = []

    def register(
        self,
        predicate: Callable[[str], bool],
        handler: Callable[[str], str],
    ) -> None:
        self._routes.append((predicate, handler))

    def dispatch(self, lower_text: str, original_text: str) -> Optional[str]:
        for predicate, handler in self._routes:
            if predicate(lower_text):
                return handler(original_text)
        return None


ROUTER = FunctionRouter()
MEMORY = ConversationMemory(max_turns=6, summary_trigger=3)


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

# Wake words to listen for. ``wakeword_utils`` falls back to bundled
# openWakeWord models when a custom ``rex.onnx`` is not present.
WAKEWORD = os.getenv("REX_WAKEWORD", "rex")
WAKEWORDS_RAW = os.getenv("REX_WAKEWORDS")
if WAKEWORDS_RAW:
    WAKEWORDS = [word.strip() for word in WAKEWORDS_RAW.split(",") if word.strip()]
else:
    WAKEWORDS = [WAKEWORD]

WAKEWORD_THRESHOLD = float(os.getenv("REX_WAKEWORD_THRESHOLD", "0.5"))

# Relative path to the wake confirmation sound.  The tone is generated in
# the repository’s ``assets`` directory on demand so no binary files need to
# live in Git.  If you place the script elsewhere, adjust this path
# accordingly.
WAKE_SOUND_PATH = ensure_wake_acknowledgment_sound()

# Mapping of user names to speaker reference WAV files.  Entries are loaded
# from the ``Memory`` directory so setting ``REX_ACTIVE_USER`` automatically
# selects the correct profile when the assistant starts.
SPEAKER_VOICES = {
    user: extract_voice_reference(profile, user_key=user)
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
    You are Rex, a focused AI voice assistant. Keep responses concise,
    polite, and grounded in the conversation. When structured tool output
    appears in the chat history, incorporate it into your reply instead of
    guessing. If you do not know an answer, say so.
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


def _build_messages(user_text: str) -> list[dict[str, str]]:
    return MEMORY.build_messages(
        persona=ASSISTANT_PERSONA,
        profile_context=PROFILE_CONTEXT,
        user_text=user_text,
    )


def _interactive_device_selection() -> Optional[int]:
    """Offer a simple terminal prompt to pick the input device."""

    auto_choice = os.getenv("REX_INPUT_DEVICE")
    try:
        devices = sd.query_devices()
    except Exception as exc:  # pragma: no cover - hardware specific
        logger.warning("Could not enumerate audio devices: %s", exc)
        return None

    if not devices:
        logger.warning("No audio input devices detected.")
        return None

    default_input = None
    try:
        default_device = sd.default.device
        if isinstance(default_device, (list, tuple)) and default_device:
            default_input = default_device[0]
    except Exception:  # pragma: no cover - defensive
        default_input = None

    if auto_choice:
        choice = auto_choice.strip()
    else:
        print("\nAvailable input devices:")
        for idx, device in enumerate(devices):
            marker = " (default)" if idx == default_input else ""
            print(f"  [{idx}] {device['name']}{marker}")
        prompt = f"Select input device [{default_input if default_input is not None else 'default'}]: "
        try:
            choice = input(prompt)
        except EOFError:  # pragma: no cover - non-interactive
            choice = ""

    choice = (choice or "").strip()
    if not choice:
        return default_input

    try:
        index = int(choice)
        if 0 <= index < len(devices):
            return index
    except ValueError:
        lowered = choice.lower()
        for idx, device in enumerate(devices):
            if lowered in device["name"].lower():
                return idx

    logger.warning("Unrecognised device selection '%s'; using default.", choice)
    return default_input


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
        logger.warning("Could not play sound '%s': %s", path, exc)


def generate_response(text: str) -> str:
    """Generate a natural-language reply using the configured language model."""

    messages = _build_messages(text)
    try:
        completion = LLM.generate(messages=messages)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("LLM error: %s", exc)
        return f"I heard: {text}. Could you repeat that while I restart my thoughts?"
    return completion


def _handle_time_command(_: str) -> str:
    now = datetime.now()
    return now.strftime("It's %I:%M %p on %A, %B %d, %Y.")


def _handle_date_command(_: str) -> str:
    today = datetime.now()
    return today.strftime("Today is %A, %B %d, %Y.")


def _handle_weather_command(original: str) -> str:
    location = original.split("weather", 1)[-1].strip() or "your area"
    if search_web is not None:
        query = f"current weather {location}"
        try:
            result = search_web(query)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Weather lookup failed: %s", exc)
        else:
            if isinstance(result, dict):
                summary = result.get("summary") or result.get("answer")
                if summary:
                    return f"Here's the latest for {location}: {summary}"
            elif isinstance(result, str) and result.strip():
                return result
    return (
        "I can check the weather once the search plugin is configured. "
        "Set up the plugin or provide a dedicated weather integration."
    )


ROUTER.register(lambda text: text.startswith("time"), _handle_time_command)
ROUTER.register(lambda text: text.startswith("date"), _handle_date_command)
ROUTER.register(lambda text: text.startswith("weather"), _handle_weather_command)


def handle_command(text: str) -> str:
    """Handle a transcribed command and return a response."""

    lower = text.strip().lower()
    routed = ROUTER.dispatch(lower, text)
    if routed is not None:
        return routed

    if lower.startswith("search ") and search_web is not None:
        query = text[7:].strip()
        result = search_web(query)
        return result or "No result found."
    return generate_response(text)


def speak(text: str, user: Optional[str] = None) -> None:
    """Convert text to speech and play it."""

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
        logger.info("Active user: %s (%s)", ACTIVE_USER_DISPLAY, ACTIVE_USER)
    else:
        logger.info("Active user: %s", ACTIVE_USER)

    selected_device = _interactive_device_selection()
    if selected_device is not None:
        try:
            current_defaults = sd.default.device
            output_device = current_defaults[1] if isinstance(current_defaults, (list, tuple)) else None
        except Exception:  # pragma: no cover - defensive
            output_device = None
        sd.default.device = (selected_device, output_device)
        logger.info("Using input device #%s", selected_device)

    wake_model, wake_keyword = load_wakeword_model(keyword=WAKEWORDS)
    logger.info("Wake phrases active: %s", wake_keyword)

    logger.info("Loading Whisper model '%s'…", WHISPER_MODEL_NAME)
    stt_model = whisper.load_model(WHISPER_MODEL_NAME)
    logger.info("Whisper loaded.")

    sample_rate = 16000
    block_size = sample_rate  # 1 second blocks
    processing_event = threading.Event()

    def process_interaction() -> None:
        try:
            logger.info("Wake word detected; acknowledging user.")
            play_sound(WAKE_SOUND_PATH)
            logger.info("Recording for %s seconds…", COMMAND_DURATION)
            recording = sd.rec(
                int(COMMAND_DURATION * sample_rate),
                samplerate=sample_rate,
                channels=1,
            )
            sd.wait()
            command_audio = recording[:, 0]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, command_audio, sample_rate)
                audio_path = tmp.name
            try:
                result = stt_model.transcribe(audio_path)
            finally:
                try:
                    os.remove(audio_path)
                except OSError:  # pragma: no cover - best effort cleanup
                    logger.debug("Could not remove temporary audio '%s'", audio_path)
            text = result.get("text", "").strip()
            if not text:
                logger.info("No speech detected after wake word.")
                return
            logger.info("Heard: %s", text)
            reply = handle_command(text)
            logger.info("Reply: %s", reply)
            speak(reply)
            MEMORY.add_turn(text, reply)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Error during interaction: %s", exc)
        finally:
            processing_event.clear()

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:  # pragma: no cover - hardware dependent
            logger.warning("Audio stream status: %s", status)
        if processing_event.is_set():
            return
        audio_data = np.squeeze(indata)
        try:
            triggered = detect_wakeword(
                wake_model,
                audio_data,
                threshold=WAKEWORD_THRESHOLD,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Wake-word detection error: %s", exc)
            return
        if triggered:
            logger.info("Wake word detected. Preparing to record.")
            processing_event.set()
            threading.Thread(target=process_interaction, daemon=True).start()

    logger.info("Rex Assistant is listening. Say the wake word to begin.")
    with sd.InputStream(
        channels=1,
        samplerate=sample_rate,
        blocksize=block_size,
        callback=audio_callback,
    ):
        try:
            input("Press Enter to stop listening…\n")
        except (KeyboardInterrupt, EOFError):
            logger.info("Stopping assistant.")


if __name__ == "__main__":
    main()
