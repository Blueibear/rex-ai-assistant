"""Async voice assistant loop orchestrating wake word, STT, LLM, and TTS."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import sys
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from wake_acknowledgment import ensure_wake_acknowledgment_sound

from .assistant_errors import AudioDeviceError, SpeechToTextError, TextToSpeechError
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
    if _import_optional("TTS") is None:
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


logger = logging.getLogger(__name__)

# Safe runtime alias: resolves to np.ndarray when numpy is available, Any otherwise.
# Module-level type alias assignments are evaluated eagerly (even with
# `from __future__ import annotations`), so we must not reference np.ndarray
# unconditionally — np is None when numpy is not installed.
_NDArray = np.ndarray if np is not None else Any

RecorderCallable = Callable[[float], Awaitable[_NDArray] | _NDArray]  # type: ignore[operator, valid-type]
IdentifySpeakerCallable = Callable[[_NDArray], str | None] | Callable[[], str | None]  # type: ignore[operator, valid-type]

# Sentence-boundary pattern for streaming TTS sentence splitting.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _split_into_sentences(text: str) -> list[str]:
    """Split *text* into sentence-sized chunks for streaming TTS."""
    sentences = _SENTENCE_BOUNDARY.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


async def _sentence_stream(text: str) -> AsyncIterator[str]:
    """Yield sentences from *text* as an async iterator."""
    for sentence in _split_into_sentences(text):
        yield sentence


@dataclass
class SynthesizedAudio:
    """Container for synthesized audio data."""

    data: np.ndarray  # type: ignore[name-defined]
    sample_rate: int


class AsyncMicrophone:
    """Async microphone recording."""

    def __init__(
        self,
        *,
        sample_rate: int,
        detection_seconds: float,
        capture_seconds: float,
        recorder: RecorderCallable | None = None,  # type: ignore[valid-type]
    ) -> None:
        self.sample_rate = sample_rate
        self._detection_seconds = detection_seconds
        self._capture_seconds = capture_seconds
        self._recorder = recorder

    async def detection_frame(self) -> np.ndarray:  # type: ignore[name-defined]
        """Record a short frame for wake word detection."""
        return await self._record(self._detection_seconds)

    async def record_phrase(self, duration: float | None = None) -> np.ndarray:  # type: ignore[name-defined]
        """Record user speech after wake word."""
        return await self._record(duration or self._capture_seconds)

    async def _record(self, duration: float) -> np.ndarray:  # type: ignore[name-defined]
        """Internal recording method."""
        np = _require_numpy()
        if duration <= 0:
            raise AudioDeviceError("Recording duration must be positive")

        if self._recorder is not None:
            result = self._recorder(duration)
            if asyncio.iscoroutine(result):
                result = await result
            return np.asarray(result, dtype=np.float32).reshape(-1)

        sd = _require_sounddevice()

        frames = max(int(self.sample_rate * duration), 1)

        def _capture() -> np.ndarray:  # type: ignore[name-defined]
            recording = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="float32")
            sd.wait()
            return recording.reshape(-1)

        try:
            data = await asyncio.to_thread(_capture)
        except Exception as exc:
            raise AudioDeviceError(str(exc)) from exc
        return np.asarray(data, dtype=np.float32)


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

    def __init__(self, model_name: str, device: str) -> None:
        whisper_module = _lazy_import_whisper()
        if whisper_module is None:
            raise SpeechToTextError("openai-whisper is not installed")

        try:
            self._model = whisper_module.load_model(model_name, device=device)
        except Exception as exc:
            raise SpeechToTextError(str(exc)) from exc

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:  # type: ignore[name-defined]
        """Transcribe audio to text."""

        def _transcribe() -> str:
            result = self._model.transcribe(audio, language="en", fp16=False)
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
        if self._provider == "xtts":
            tts_class = _lazy_import_tts()
        else:
            tts_class = None

        if self._provider == "xtts" and tts_class is not None:
            try:
                torch = import_module("torch")
                self._tts = tts_class(
                    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                    progress_bar=False,
                )
                if torch.cuda.is_available():
                    self._tts.to("cuda")
            except Exception as exc:
                logger.warning("XTTS init failed: %s", exc)

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
        if self._tts is None:
            raise TextToSpeechError("XTTS not initialized")
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
            def _synthesize(_chunk=chunk, _chunk_path=chunk_path) -> None:
                self._tts.tts_to_file(  # type: ignore[union-attr]
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
            with suppress(FileNotFoundError):
                Path(chunk_path).unlink()

    async def speak_streaming(
        self,
        sentences: AsyncIterator[str],  # type: ignore[type-arg]
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
                # Convert mp3 to wav for playback
                data, rate = sf.read(output_path)
                wav_path = output_path.replace(".mp3", ".wav")
                sf.write(wav_path, data, rate)

                def _play() -> None:
                    wave_obj = sa.WaveObject.from_wave_file(wav_path)
                    play_obj = wave_obj.play()
                    play_obj.wait_done()

                await asyncio.to_thread(_play)
                with suppress(FileNotFoundError):
                    Path(wav_path).unlink()
        finally:
            with suppress(FileNotFoundError):
                Path(output_path).unlink()

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
        acknowledge: Callable[[], Awaitable[None]] | None = None,
        identify_speaker: IdentifySpeakerCallable | None = None,  # type: ignore[valid-type]
    ) -> None:
        self._assistant = assistant
        self._wake_listener = wake_listener
        self._detection_source = detection_source
        self._record_phrase = record_phrase
        self._transcribe = transcribe
        self._speak = speak
        self._speak_streaming = speak_streaming
        self._acknowledge = acknowledge
        self._identify_speaker = identify_speaker
        self._identify_speaker_accepts_audio = self._resolve_identify_speaker_signature(
            identify_speaker
        )

    @staticmethod
    def _resolve_identify_speaker_signature(
        identify_speaker: IdentifySpeakerCallable | None,  # type: ignore[valid-type]
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

    async def run(self, max_interactions: int | None = None) -> None:
        """Run the voice loop for a specified number of interactions."""
        from .voice_latency import VoiceLatencyTracker  # noqa: PLC0415

        interactions = 0

        try:
            async for _ in self._wake_listener.listen(self._detection_source):
                try:
                    tracker = VoiceLatencyTracker()

                    if self._acknowledge:
                        await self._acknowledge()

                    # Record user speech
                    audio = await self._record_phrase()

                    # Optionally identify the speaker from voice
                    if self._identify_speaker is not None:
                        try:
                            if self._identify_speaker_accepts_audio:
                                self._identify_speaker(audio)
                            else:
                                self._identify_speaker()
                        except Exception as exc:
                            logger.warning("Voice identity check failed: %s", exc)

                    # Transcribe to text
                    tracker.mark("stt_start")
                    transcript = await self._transcribe(audio)
                    tracker.mark("stt_end")
                    if not transcript:
                        logger.info("No speech detected")
                        continue

                    # Get LLM response
                    tracker.mark("llm_start")
                    response = await self._assistant.generate_reply(transcript)
                    tracker.mark("llm_end")

                    # Ensure response ends with period for better TTS
                    if response and not response.endswith("."):
                        response = response + "."

                    # Speak response — use streaming path if available
                    tracker.mark("tts_synthesis_start")
                    if self._speak_streaming is not None:
                        tracker.mark("tts_first_chunk")
                        await self._speak_streaming(_sentence_stream(response))
                    else:
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


def _build_voice_id_callback() -> IdentifySpeakerCallable | None:  # type: ignore[valid-type]
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

    def _identify(audio: _NDArray) -> str | None:  # type: ignore[valid-type]
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
    device: str = "cpu",
    language: str = "en",
    speaker_wav: str | None = None,
    wake_sound_path: Path | None = None,
) -> VoiceLoop:
    """Build a VoiceLoop with default components.

    When ``voice_identity.enabled=true`` is set in ``config/rex_config.json``
    and at least one user is enrolled, an ``identify_speaker`` callback is
    built and wired into the voice loop automatically.
    """
    from .wakeword.listener import build_default_detector

    mic = AsyncMicrophone(
        sample_rate=sample_rate,
        detection_seconds=detection_seconds,
        capture_seconds=capture_seconds,
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
        speak_streaming=lambda sentences: tts.speak_streaming(
            sentences, speaker_wav=speaker_wav
        ),
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
