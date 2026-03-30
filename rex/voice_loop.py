"""Async voice assistant loop orchestrating wake word, STT, LLM, and TTS.

RELATIONSHIP NOTE — two voice_loop files exist in this repo:
- ``rex/voice_loop.py`` (this file, package): canonical implementation.
  ``rex_loop.py`` imports ``build_voice_loop`` from here and this is the
  authoritative voice loop used when Rex starts.
- ``voice_loop.py`` (repo root): legacy implementation containing
  ``AsyncRexAssistant``. Kept for backward compatibility only. Changes here
  do NOT affect the ``rex_loop.py`` startup path.
"""

from __future__ import annotations

import asyncio
import io
import inspect
import logging
import os
import re
import sys
import tempfile
import wave
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union, cast

from typing_extensions import TypeAlias

from wake_acknowledgment import ensure_wake_acknowledgment_sound

from .assistant_errors import AudioDeviceError, AudioFormatError, SpeechToTextError, TextToSpeechError
from .config import settings
from .memory import (
    extract_voice_reference,
    load_all_profiles,
    load_users_map,
    resolve_user_key,
)
from .tts_utils import chunk_text_for_xtts


def _import_optional(module_name: str):
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    if find_spec(module_name) is None:
        return None
    return import_module(module_name)


def _lazy_import_numpy():
    return _import_optional("numpy")


np = _lazy_import_numpy()


def _lazy_import_simpleaudio():
    return _import_optional("simpleaudio")


sa = _lazy_import_simpleaudio()
sd = None


def _lazy_import_whisper():
    return _import_optional("whisper")


def _lazy_import_tts():
    # Only check availability - do NOT import TTS yet (it triggers
    # internal transformers imports that need the shim first).
    if find_spec("TTS") is None:
        return None
    from rex.compat import ensure_transformers_compatibility

    ensure_transformers_compatibility()
    return import_module("TTS.api").TTS


def _lazy_import_soundfile():
    return _import_optional("soundfile")


def _load_sounddevice():
    global sd
    if sd is not None:
        return sd
    sd = _import_optional("sounddevice")
    return sd


def _require_numpy():
    if np is None:
        raise AudioDeviceError("numpy is required for audio processing")
    return np


def _require_sounddevice():
    module = _load_sounddevice()
    if module is None:
        raise AudioDeviceError("sounddevice is not installed")
    return module


def _device_name(device: Any) -> str:
    if isinstance(device, dict):
        return str(device.get("name", "<unknown>"))
    return str(getattr(device, "name", "<unknown>"))


def _max_input_channels(device: Any) -> int:
    if isinstance(device, dict):
        value = device.get("max_input_channels", 0)
    else:
        value = getattr(device, "max_input_channels", 0)
    return int(value or 0)


def _available_input_devices(devices: Any) -> list[str]:
    available: list[str] = []
    for index, device in enumerate(devices):
        if _max_input_channels(device) > 0:
            available.append(f"{index}: {_device_name(device)}")
    return available


def _validate_input_device_index(device_index: int | None) -> int | None:
    if device_index is None:
        return None

    sd_module = _require_sounddevice()
    try:
        devices = sd_module.query_devices()
    except Exception as exc:
        raise AudioDeviceError(str(exc)) from exc

    available_devices = _available_input_devices(devices)
    available_list = ", ".join(available_devices) if available_devices else "none"

    try:
        device = devices[device_index]
    except (IndexError, KeyError, TypeError):
        raise AudioDeviceError(
            f"Input device {device_index} not found. Available: {available_list}"
        ) from None

    if _max_input_channels(device) <= 0:
        raise AudioDeviceError(
            f"Input device {device_index} not found. Available: {available_list}"
        )

    return device_index


def _detect_audio_format(audio_buffer: bytes) -> str:
    header = audio_buffer[:4]
    if not header:
        return "empty"
    if header.startswith(b"ID3"):
        return "ID3"
    text = header.decode("ascii", errors="ignore")
    text = "".join(char for char in text if char.isprintable()).strip()
    return text or header.hex()


def _to_wav_buffer(audio: AudioArray | bytes | bytearray | memoryview, sample_rate: int) -> bytes:
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return bytes(audio)

    numpy = _require_numpy()
    samples = numpy.asarray(audio, dtype=numpy.float32).reshape(-1)
    samples = numpy.clip(samples, -1.0, 1.0)
    pcm16 = (samples * 32767).astype(numpy.int16)

    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16.tobytes())
        return buffer.getvalue()


logger = logging.getLogger(__name__)
_USE_CONFIG_LANGUAGE = object()

if TYPE_CHECKING:
    from numpy.typing import NDArray

    AudioArray: TypeAlias = NDArray[Any]
else:
    AudioArray: TypeAlias = Any

RecorderCallable = Callable[[float], Union[Awaitable[AudioArray], AudioArray]]
IdentifySpeakerCallable = Union[Callable[[AudioArray], Optional[str]], Callable[[], Optional[str]]]

# Backwards-compatible runtime alias used by optional-import tests.
_NDArray = np.ndarray if np is not None else Any


# Sentence-boundary pattern for streaming TTS sentence splitting.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# Short phrase used to pre-warm the TTS engine on startup.
_WARMUP_PHRASE = "."

# Common single-word abbreviations that should not trigger sentence boundaries.
# Matched as whole words (case-insensitive) followed by "." and whitespace.
_ABBREV_WORDS: frozenset[str] = frozenset(
    [
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "vs",
        "etc",
        "al",
        "st",
        "fig",
        "dept",
        "est",
        "approx",
        "cf",
        "rev",
        "gen",
        "col",
        "lt",
        "sgt",
        "capt",
        "gov",
        "sen",
        "rep",
        "no",
        "vol",
        "ave",
        "blvd",
    ]
)

# Abbreviations containing internal dots (e.g., i.e.) followed by "." and whitespace.
_ABBREV_DOT: frozenset[str] = frozenset(
    ["e.g", "i.e", "a.m", "p.m", "u.s", "u.k", "u.n"]
)

# Placeholder character used to protect abbreviation periods during splitting.
_ABBREV_PLACEHOLDER = "\x00"


def _protect_abbreviations(text: str) -> str:
    """Replace trailing periods in known abbreviations with a placeholder.

    This prevents *_SENTENCE_BOUNDARY* from treating abbreviations like
    "Dr.", "Mr.", or "e.g." as sentence-ending punctuation.  Original
    casing is preserved via a capturing group in each substitution.
    """
    protected = text
    # Single-word abbreviations: word-boundary + abbr + "." + whitespace.
    # Group 1 captures the original-cased abbreviation so it is preserved.
    for abbr in _ABBREV_WORDS:
        protected = re.sub(
            rf"(?<!\w)({re.escape(abbr)})\.\s",
            r"\1" + _ABBREV_PLACEHOLDER + " ",
            protected,
            flags=re.IGNORECASE,
        )
    # Dot-internal abbreviations: abbr + "." + whitespace
    for abbr in _ABBREV_DOT:
        protected = re.sub(
            rf"({re.escape(abbr)})\.\s",
            r"\1" + _ABBREV_PLACEHOLDER + " ",
            protected,
            flags=re.IGNORECASE,
        )
    return protected


def _split_into_sentences(text: str) -> list[str]:
    """Split *text* into sentence-sized chunks for streaming TTS.

    Uses NLTK ``sent_tokenize`` when available; otherwise falls back to an
    abbreviation-aware regex splitter that does not break on common titles
    (Dr., Mr.) or abbreviations (e.g., etc.).
    """
    stripped = text.strip()
    if not stripped:
        return []

    # Try NLTK sent_tokenize first (handles abbreviations natively).
    if find_spec("nltk") is not None:
        try:
            nltk = _import_optional("nltk")
            if nltk is None:
                raise ImportError("nltk is not available")

            sentences = nltk.sent_tokenize(stripped)
            return [s.strip() for s in sentences if s.strip()]
        except Exception:
            # punkt tokenizer not downloaded or other NLTK error — fall through.
            pass

    # Abbreviation-aware regex fallback.
    protected = _protect_abbreviations(stripped)
    parts = _SENTENCE_BOUNDARY.split(protected)
    return [s.replace(_ABBREV_PLACEHOLDER, ".").strip() for s in parts if s.strip()]


async def _sentence_stream(text: str) -> AsyncIterator[str]:
    """Yield sentences from *text* as an async iterator."""
    for sentence in _split_into_sentences(text):
        yield sentence


def _extract_completed_sentences(buffer: str) -> tuple[list[str], str]:
    """Return completed sentences and the remaining partial buffer."""
    protected = _protect_abbreviations(buffer)
    matches = list(_SENTENCE_BOUNDARY.finditer(protected))
    if not matches:
        return [], buffer

    split_index = matches[-1].end()
    completed_text = buffer[:split_index]
    remainder = buffer[split_index:]
    return _split_into_sentences(completed_text), remainder


async def _sentence_buffer_stream(tokens: AsyncIterator[str]) -> AsyncIterator[str]:
    """Convert a token stream into sentence chunks for streaming TTS."""
    buffer = ""
    async for token in tokens:
        if not token:
            continue
        buffer += token
        sentences, buffer = _extract_completed_sentences(buffer)
        for sentence in sentences:
            yield sentence

    for sentence in _split_into_sentences(buffer):
        yield sentence


@dataclass
class SynthesizedAudio:
    """Container for synthesized audio data."""

    data: AudioArray
    sample_rate: int


class AsyncMicrophone:
    """Async microphone recording."""

    def __init__(
        self,
        *,
        sample_rate: int,
        detection_seconds: float,
        capture_seconds: float,
        device_index: int | None = None,
        recorder: RecorderCallable | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self._detection_seconds = detection_seconds
        self._capture_seconds = capture_seconds
        self._device_index = device_index
        self._recorder = recorder

    async def detection_frame(self) -> AudioArray:
        """Record a short frame for wake word detection."""
        return await self._record(self._detection_seconds)

    async def record_phrase(self, duration: float | None = None) -> AudioArray:
        """Record user speech after wake word."""
        return await self._record(duration or self._capture_seconds)

    async def _record(self, duration: float) -> AudioArray:
        """Internal recording method."""
        np = _require_numpy()
        if duration <= 0:
            raise AudioDeviceError("Recording duration must be positive")

        if self._recorder is not None:
            result = self._recorder(duration)
            if asyncio.iscoroutine(result):
                result = await result
            return cast(AudioArray, np.asarray(result, dtype=np.float32).reshape(-1))

        sd = _require_sounddevice()

        frames = max(int(self.sample_rate * duration), 1)

        def _capture() -> np.ndarray:  # type: ignore[name-defined]
            recording = sd.rec(
                frames,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self._device_index,
            )
            sd.wait()
            return recording.reshape(-1)

        try:
            data = await asyncio.to_thread(_capture)
        except Exception as exc:
            raise AudioDeviceError(str(exc)) from exc
        return cast(AudioArray, np.asarray(data, dtype=np.float32))


class WakeAcknowledgement:
    """Play acknowledgement sound when wake word is detected."""

    def __init__(self, sound_path: Path | None = None) -> None:
        default_path = Path(__file__).resolve().parents[1] / "assets" / "wake_acknowledgment.wav"
        self._sound_path = Path(sound_path) if sound_path else default_path
        if not self._sound_path.exists():
            try:
                ensure_wake_acknowledgment_sound(path=str(self._sound_path))
            except Exception as exc:
                logger.warning("Failed to generate wake acknowledgment sound: %s", exc)

    async def play(self) -> None:
        """Play the wake acknowledgement sound."""
        if not self._sound_path.exists():
            return
        if sa is None and _load_sounddevice() is None:
            logger.warning("No audio playback backend available for wake acknowledgment.")
            return

        def _play() -> None:
            if sa is not None:
                wave_obj = sa.WaveObject.from_wave_file(str(self._sound_path))
                play_obj = wave_obj.play()
                play_obj.wait_done()
                return
            sd = _require_sounddevice()
            sf = _lazy_import_soundfile()
            if sf is None:
                raise AudioDeviceError("soundfile is required for wake acknowledgement playback")
            data, rate = sf.read(str(self._sound_path), dtype="float32")
            sd.play(data, rate)
            sd.wait()

        try:
            await asyncio.to_thread(_play)
        except Exception as exc:
            logger.warning("Wake acknowledgement failed: %s", exc)


class SpeechToText:
    """Speech-to-text using Whisper."""

    def __init__(
        self,
        model_name: str,
        device: str,
        language: str | None | object = _USE_CONFIG_LANGUAGE,
    ) -> None:
        whisper_module = _lazy_import_whisper()
        if whisper_module is None:
            raise SpeechToTextError("openai-whisper is not installed")

        if language is _USE_CONFIG_LANGUAGE:
            language = getattr(settings, "whisper_language", "en")
        self._language = cast(Optional[str], language)

        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        try:
            self._model = whisper_module.load_model(model_name, device=device)
        except Exception as exc:
            raise SpeechToTextError(str(exc)) from exc

    async def transcribe(
        self,
        audio: AudioArray | bytes | bytearray | memoryview,
        sample_rate: int,
    ) -> str:
        """Transcribe audio to text."""

        audio_buffer = _to_wav_buffer(audio, sample_rate)
        if audio_buffer[:4] != b"RIFF":
            detected_format = _detect_audio_format(audio_buffer)
            raise AudioFormatError(f"Expected WAV, got {detected_format}")

        def _transcribe() -> str:
            result = self._model.transcribe(audio, language=self._language, fp16=False)
            return str(result.get("text", "")).strip()

        try:
            return await asyncio.to_thread(_transcribe)
        except Exception as exc:
            logger.error("[STT] Whisper failed: %s", exc)
            raise SpeechToTextError(str(exc)) from exc


class TextToSpeech:
    """Text-to-speech synthesis."""

    def __init__(self, *, language: str, default_speaker: str | None = None) -> None:
        self._language = language
        self._default_speaker = default_speaker
        self._tts_speed = getattr(settings, "tts_speed", 1.08)

        # Get TTS settings from config (defaults: xtts provider, en-US-AndrewNeural voice)
        self._provider = getattr(settings, "tts_provider", "xtts").lower()
        self._edge_voice = getattr(settings, "tts_voice", None) or "en-US-AndrewNeural"

        self._tts = None
        self._xtts_init_error: str | None = None
        if self._provider == "xtts":
            self._initialize_xtts()

    def _initialize_xtts(self) -> bool:
        """Initialize XTTS model, storing diagnostics on failure."""
        if self._tts is not None:
            return True

        tts_class = _lazy_import_tts()
        if tts_class is None:
            self._xtts_init_error = "Coqui XTTS is not installed"
            logger.warning("XTTS init skipped: %s", self._xtts_init_error)
            return False

        try:
            torch = import_module("torch")
            self._tts = tts_class(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
            )
            if torch.cuda.is_available() and self._tts is not None:
                self._tts.to("cuda")
            self._xtts_init_error = None
            return True
        except Exception as exc:
            self._xtts_init_error = str(exc)
            logger.warning("XTTS init failed: %s", exc)
            return False

    async def speak(self, text: str, *, speaker_wav: str | None = None) -> None:
        """Synthesize and play text as speech."""
        if not text:
            return

        text = self._clean_text(text)
        if not text:
            return

        try:
            if self._provider == "xtts":
                await self._speak_xtts(text, speaker_wav)
            elif self._provider == "edge":
                await self._speak_edge(text)
            elif self._provider == "windows":
                await self._speak_windows(text)
            else:
                print(f"Rex: {text}")
        except Exception as exc:
            logger.error("[TTS] Failed: %s", exc)
            print(f"Rex: {text}")

    def _clean_text(self, text: str) -> str:
        """Clean text for TTS."""
        if "Additional info:" in text:
            text = text.split("Additional info:")[0].strip()
        text = re.sub(r"http[s]?://\S+", "", text)
        text = re.sub(r"uddg=\S+", "", text)
        text = re.sub(r"\[.*?\]", "", text)
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        text = ". ".join(sentences[:2])
        return text + "." if text and not text.endswith(".") else text

    async def _speak_xtts(self, text: str, speaker_wav: str | None) -> None:
        """Synthesize speech using XTTS, playing each chunk immediately."""
        if self._tts is None and not self._initialize_xtts():
            reason = self._xtts_init_error or "unknown initialization error"
            raise TextToSpeechError(f"XTTS not initialized: {reason}")
        sf = _lazy_import_soundfile()
        if sf is None:
            raise TextToSpeechError("soundfile is required for XTTS output")
        chunks = chunk_text_for_xtts(text, max_tokens=300)
        if not chunks:
            return

        for chunk in chunks:
            await self._synthesize_and_play_chunk(chunk, speaker_wav, sf)

    async def _synthesize_and_play_chunk(
        self, chunk: str, speaker_wav: str | None, sf: Any
    ) -> None:
        """Synthesize a single text chunk and play it immediately."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            chunk_path = tmp.name

        try:

            tts_engine = self._tts
            if tts_engine is None:
                raise TextToSpeechError("XTTS not initialized")

            def _synthesize(_chunk=chunk, _chunk_path=chunk_path) -> None:
                tts_engine.tts_to_file(
                    text=_chunk,
                    speaker_wav=speaker_wav or self._default_speaker,
                    language=self._language,
                    file_path=_chunk_path,
                    speed=self._tts_speed,
                )

            await asyncio.to_thread(_synthesize)

            if sa is not None and Path(chunk_path).exists():

                def _play(_path=chunk_path) -> None:
                    wave_obj = sa.WaveObject.from_wave_file(_path)
                    play_obj = wave_obj.play()
                    play_obj.wait_done()

                await asyncio.to_thread(_play)
        finally:
            try:
                os.unlink(chunk_path)
            except (OSError, PermissionError) as exc:
                logger.warning("Failed to remove temp file %s: %s", chunk_path, exc)

    async def warmup(self, *, speaker_wav: str | None = None) -> None:
        """Pre-warm the TTS engine by synthesizing a short phrase in the background.

        Call via ``asyncio.create_task(tts.warmup())`` so it does not block startup.
        """
        try:
            logger.info("[TTS] Pre-warming engine...")
            await self.speak(_WARMUP_PHRASE, speaker_wav=speaker_wav)
            logger.info("[TTS] Pre-warm complete.")
        except Exception as exc:
            logger.warning("[TTS] Pre-warm failed (non-fatal): %s", exc)

    async def speak_streaming(
        self,
        sentences: AsyncIterator[str],
        *,
        speaker_wav: str | None = None,
    ) -> None:
        """Speak each sentence from an async iterator as soon as it arrives.

        This enables first audio to begin playing before the full response is
        available, reducing perceived latency.
        """
        try:
            async for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                try:
                    await self.speak(sentence, speaker_wav=speaker_wav)
                except Exception as exc:
                    logger.error("[TTS streaming] chunk failed: %s", exc)
        except Exception as exc:
            logger.error("[TTS streaming] failed: %s", exc)

    async def _speak_edge(self, text: str) -> None:
        """Synthesize speech using Edge TTS."""
        try:
            import edge_tts
        except ImportError:
            raise TextToSpeechError("edge-tts is not installed")
        sf = _lazy_import_soundfile()
        if sf is None:
            raise TextToSpeechError("soundfile is required for Edge TTS playback")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            output_path = tmp.name

        try:
            communicate = edge_tts.Communicate(text, self._edge_voice)
            await communicate.save(output_path)

            if sa is not None and Path(output_path).exists():
                # Convert mp3 to wav and play — all blocking I/O in thread executor.
                wav_path = output_path.replace(".mp3", ".wav")

                def _convert_and_play(_src=output_path, _dst=wav_path) -> None:
                    data, rate = sf.read(_src)
                    sf.write(_dst, data, rate)
                    wave_obj = sa.WaveObject.from_wave_file(_dst)
                    play_obj = wave_obj.play()
                    play_obj.wait_done()

                try:
                    await asyncio.to_thread(_convert_and_play)
                finally:
                    try:
                        os.unlink(wav_path)
                    except (OSError, PermissionError) as exc:
                        logger.warning("Failed to remove temp file %s: %s", wav_path, exc)
        finally:
            try:
                os.unlink(output_path)
            except (OSError, PermissionError) as exc:
                logger.warning("Failed to remove temp file %s: %s", output_path, exc)

    async def _speak_windows(self, text: str) -> None:
        """Synthesize speech using Windows SAPI."""
        try:
            import pyttsx3
        except ImportError:
            raise TextToSpeechError("pyttsx3 is not installed")

        def _speak() -> None:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()

        await asyncio.to_thread(_speak)


class VoiceLoop:
    """Main voice assistant loop coordinating wake word, STT, LLM, and TTS."""

    def __init__(
        self,
        assistant,
        *,
        wake_listener,
        detection_source: Callable[[], Awaitable[np.ndarray]],  # type: ignore[name-defined]
        record_phrase: Callable[[], Awaitable[np.ndarray]],  # type: ignore[name-defined]
        transcribe: Callable[[np.ndarray], Awaitable[str]],  # type: ignore[name-defined]
        speak: Callable[[str], Awaitable[None]],
        speak_streaming: Callable[[AsyncIterator[str]], Awaitable[None]] | None = None,
        warmup: Callable[[], Awaitable[None]] | None = None,
        acknowledge: Callable[[], Awaitable[None]] | None = None,
        identify_speaker: IdentifySpeakerCallable | None = None,
    ) -> None:
        self._assistant = assistant
        if getattr(settings, "use_openclaw_voice_backend", False):
            try:
                from rex.openclaw.voice_bridge import VoiceBridge

                self._assistant = VoiceBridge()
                logger.info("Voice loop using OpenClaw VoiceBridge backend")
            except Exception as exc:
                logger.warning(
                    "Failed to create VoiceBridge (falling back to default assistant): %s", exc
                )
        self._wake_listener = wake_listener
        self._detection_source = detection_source
        self._record_phrase = record_phrase
        self._transcribe = transcribe
        self._speak = speak
        self._speak_streaming = speak_streaming
        self._warmup = warmup
        self._acknowledge = acknowledge
        self._identify_speaker = identify_speaker
        self._identify_speaker_accepts_audio = self._resolve_identify_speaker_signature(
            identify_speaker
        )

    @staticmethod
    def _resolve_identify_speaker_signature(
        identify_speaker: IdentifySpeakerCallable | None,
    ) -> bool:
        """Return True when identify_speaker accepts an audio argument."""
        if identify_speaker is None:
            return False
        try:
            signature = inspect.signature(identify_speaker)
        except (TypeError, ValueError):
            return False

        for parameter in signature.parameters.values():
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                return True

        return False

    async def _safe_acknowledge(self) -> None:
        """Play acknowledgement tone, suppressing all errors to keep pipeline running."""
        try:
            if self._acknowledge is not None:
                await self._acknowledge()
        except Exception as exc:
            logger.warning("[Ack] Acknowledgement tone failed (non-fatal): %s", exc)

    async def warmup(self) -> None:
        """Pre-warm TTS in the background.

        Schedule as a fire-and-forget task::

            asyncio.create_task(voice_loop.warmup())
        """
        if self._warmup is not None:
            await self._warmup()

    async def run(self, max_interactions: int | None = None) -> None:
        """Run the voice loop for a specified number of interactions."""
        from .voice_latency import VoiceLatencyTracker  # noqa: PLC0415

        interactions = 0

        try:
            async for _ in self._wake_listener.listen(self._detection_source):
                try:
                    tracker = VoiceLatencyTracker()

                    # Fire acknowledgment tone concurrently with recording so the
                    # microphone starts capturing immediately after wake word.
                    # Playback failure is suppressed to keep the pipeline running.
                    if self._acknowledge:
                        asyncio.create_task(self._safe_acknowledge())

                    # Record user speech
                    audio = await self._record_phrase()

                    # Optionally identify the speaker from voice
                    if self._identify_speaker is not None:
                        try:
                            if self._identify_speaker_accepts_audio:
                                cast(Any, self._identify_speaker)(audio)
                            else:
                                cast(Any, self._identify_speaker)()
                        except Exception as exc:
                            logger.warning("Voice identity check failed: %s", exc)

                    # Transcribe to text
                    tracker.mark("stt_start")
                    transcript = await self._transcribe(audio)
                    tracker.mark("stt_end")
                    if not transcript:
                        logger.info("No speech detected")
                        continue

                    stream_reply = getattr(self._assistant, "stream_reply", None)

                    # Get LLM response - voice_mode=True enables conciseness prompt
                    tracker.mark("llm_start")
                    tracker.mark("tts_synthesis_start")
                    if self._speak_streaming is not None and callable(stream_reply):
                        tracker.mark("tts_first_chunk")
                        await self._speak_streaming(
                            _sentence_buffer_stream(stream_reply(transcript, voice_mode=True))
                        )
                        tracker.mark("llm_end")
                    else:
                        response = await self._assistant.generate_reply(transcript, voice_mode=True)
                        tracker.mark("llm_end")

                        if response and not response.endswith("."):
                            response = response + "."

                        await self._speak(response)
                    tracker.mark("tts_synthesis_end")
                    tracker.mark("playback_start")
                    tracker.log_summary()

                except SpeechToTextError as exc:
                    logger.error("STT error: %s", exc)
                    # Continue loop on transcription errors
                except AudioDeviceError as exc:
                    logger.error("Audio device error: %s", exc)
                    break
                except Exception as exc:
                    logger.error("Unexpected error in voice loop: %s", exc)

                interactions += 1
                if max_interactions is not None and interactions >= max_interactions:
                    break
        except AudioDeviceError as exc:
            logger.error("Audio device error: %s", exc)


def _build_voice_id_callback() -> IdentifySpeakerCallable | None:
    """Build an identify_speaker callback if voice identity is enabled.

    Reads the voice_identity config section, loads enrolled embeddings, and
    returns a callback that:
    - Converts a numpy audio array to PCM bytes
    - Generates an embedding via the configured backend
    - Runs recognition against all enrolled users
    - Calls resolve_speaker_identity() to update the session user

    Returns None when voice identity is disabled or no users are enrolled.
    All errors are caught and logged; the callback never raises.
    """
    try:
        from rex.config_manager import load_config as _load_json_config
        from rex.voice_identity.types import VoiceIdentityConfig

        raw_cfg = _load_json_config()
        vi_dict = raw_cfg.get("voice_identity", {})
        vi_cfg = VoiceIdentityConfig(
            enabled=vi_dict.get("enabled", False),
            accept_threshold=float(vi_dict.get("accept_threshold", 0.85)),
            review_threshold=float(vi_dict.get("review_threshold", 0.65)),
            embedding_dim=int(vi_dict.get("embedding_dim", 192)),
            model_id=str(vi_dict.get("model_id", "synthetic")),
        )
    except Exception as exc:
        logger.debug("Could not load voice_identity config: %s", exc)
        return None

    if not vi_cfg.enabled:
        return None

    try:
        from rex.voice_identity.embeddings_store import EmbeddingsStore
        from rex.voice_identity.optional_deps import get_embedding_backend
        from rex.voice_identity.recognizer import SpeakerRecognizer

        memory_dir = Path(__file__).resolve().parent.parent / "Memory"
        store = EmbeddingsStore(memory_dir)
        enrolled = store.load_all()

        if not enrolled:
            logger.info(
                "Voice identity enabled but no users are enrolled. "
                "Use 'rex voice-id enroll' to enroll users."
            )
            return None

        backend = get_embedding_backend(vi_cfg.model_id, dim=vi_cfg.embedding_dim)
        recognizer = SpeakerRecognizer(vi_cfg)

        logger.info(
            "Voice identity active: backend=%s, enrolled=%d user(s), " "accept=%.2f, review=%.2f",
            vi_cfg.model_id,
            len(enrolled),
            vi_cfg.accept_threshold,
            vi_cfg.review_threshold,
        )
    except ImportError as exc:
        logger.warning(
            "Voice identity backend unavailable: %s. "
            "Install optional extras: pip install '.[voice-id]'",
            exc,
        )
        return None
    except Exception as exc:
        logger.warning("Failed to initialise voice identity: %s", exc)
        return None

    def _identify(audio: AudioArray) -> str | None:
        try:
            # Convert numpy float32 array to raw bytes for the embedding backend
            np_mod = _lazy_import_numpy()
            if np_mod is not None:
                pcm_bytes = np_mod.asarray(audio, dtype=np_mod.float32).tobytes()
            else:
                # Fallback: use bytes() if numpy unavailable at call time
                pcm_bytes = bytes(audio)

            vector = backend.embed(pcm_bytes)
            result = recognizer.recognize(vector, enrolled)

            from rex.voice_identity.fallback_flow import resolve_speaker_identity

            resolved = resolve_speaker_identity(result)

            if result.decision.value == "recognized":
                logger.info(
                    "Voice recognized: user=%s score=%.3f",
                    result.best_user_id,
                    result.score,
                )
            elif result.decision.value == "review":
                logger.info(
                    "Voice uncertain (review): best_match=%s score=%.3f. "
                    "Run 'rex identify' to set user manually.",
                    result.best_user_id,
                    result.score,
                )

            return resolved
        except Exception as exc:
            logger.warning("Voice identity check failed: %s", exc)
            return None

    return _identify


def build_voice_loop(
    assistant,
    *,
    sample_rate: int = 16000,
    detection_seconds: float = 1.0,
    capture_seconds: float = 5.0,
    whisper_model: str = "base",
    device: str = "auto",
    language: str = "en",
    speaker_wav: str | None = None,
    wake_sound_path: Path | None = None,
) -> VoiceLoop:
    """Build a VoiceLoop with default components.

    When ``voice_identity.enabled=true`` is set in ``config/rex_config.json``
    and at least one user is enrolled, an ``identify_speaker`` callback is
    built and wired into the voice loop automatically.
    """
    input_device_index = _validate_input_device_index(settings.audio_input_device)

    from .wakeword.listener import build_default_detector

    mic = AsyncMicrophone(
        sample_rate=sample_rate,
        detection_seconds=detection_seconds,
        capture_seconds=capture_seconds,
        device_index=input_device_index,
    )

    wake_listener = build_default_detector(
        sample_rate=sample_rate,
        chunk_duration=detection_seconds,
    )

    stt = SpeechToText(model_name=whisper_model, device=device)
    tts = TextToSpeech(language=language, default_speaker=speaker_wav)
    ack = WakeAcknowledgement(sound_path=wake_sound_path)

    identify_speaker = _build_voice_id_callback()

    return VoiceLoop(
        assistant,
        wake_listener=wake_listener,
        detection_source=mic.detection_frame,
        record_phrase=mic.record_phrase,
        transcribe=lambda audio: stt.transcribe(audio, sample_rate),
        speak=lambda text: tts.speak(text, speaker_wav=speaker_wav),
        speak_streaming=lambda sentences: tts.speak_streaming(sentences, speaker_wav=speaker_wav),
        warmup=lambda: tts.warmup(speaker_wav=speaker_wav),
        acknowledge=ack.play,
        identify_speaker=identify_speaker,
    )


def _resolve_voice_reference() -> str | None:
    """Resolve voice reference for the default user.

    Returns:
        Path to voice sample file, or None if not configured
    """
    try:
        users_map = load_users_map()
        profiles = load_all_profiles()

        # Get default user
        default_user = settings.default_user or settings.user_id or "default"
        user_key = resolve_user_key(default_user, users_map, profiles=profiles)

        if not user_key:
            user_key = default_user

        # Load profile and extract voice reference
        if user_key in profiles:
            return extract_voice_reference(profiles[user_key], user_key=user_key)

        return None
    except Exception as exc:
        logger.warning("Failed to resolve voice reference: %s", exc)
        return None


__all__ = [
    "AsyncMicrophone",
    "WakeAcknowledgement",
    "SpeechToText",
    "SynthesizedAudio",
    "TextToSpeech",
    "VoiceLoop",
    "build_voice_loop",
    "_resolve_voice_reference",
]
