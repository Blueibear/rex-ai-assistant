"""Async voice assistant loop orchestrating wake word, STT, LLM, and TTS."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Awaitable, Callable, Optional

import numpy as np

from .assistant import Assistant
from .assistant_errors import (
    AudioDeviceError,
    SpeechToTextError,
    TextToSpeechError,
    WakeWordError,
)
from .config import settings
from .memory import (
    extract_voice_reference,
    load_all_profiles,
    load_users_map,
    resolve_user_key,
)
from .wakeword.listener import WakeWordListener, build_default_detector
from .wakeword.utils import load_wakeword_model

try:  # pragma: no cover - optional dependency
    import simpleaudio as sa  # type: ignore
except ImportError:  # pragma: no cover - degrade gracefully
    sa = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover - degrade gracefully
    sd = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from TTS.api import TTS  # type: ignore
except ImportError:  # pragma: no cover - degrade gracefully
    TTS = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import whisper  # type: ignore
except ImportError:  # pragma: no cover - degrade gracefully
    whisper = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

RecorderCallable = Callable[[float], Awaitable[np.ndarray] | np.ndarray]


class AsyncMicrophone:
    """Provide asynchronous helpers for recording microphone input."""

    def __init__(
        self,
        *,
        sample_rate: int,
        detection_seconds: float,
        capture_seconds: float,
        recorder: RecorderCallable | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self._detection_seconds = detection_seconds
        self._capture_seconds = capture_seconds
        self._recorder = recorder

    async def detection_frame(self) -> np.ndarray:
        return await self._record(self._detection_seconds)

    async def record_phrase(self, duration: Optional[float] = None) -> np.ndarray:
        return await self._record(duration or self._capture_seconds)

    async def _record(self, duration: float) -> np.ndarray:
        if duration <= 0:
            raise AudioDeviceError("Recording duration must be positive")

        if self._recorder is not None:
            result = self._recorder(duration)
            if asyncio.iscoroutine(result):
                result = await result
            return np.asarray(result, dtype=np.float32).reshape(-1)

        if sd is None:
            raise AudioDeviceError("sounddevice is not installed")

        frames = max(int(self.sample_rate * duration), 1)

        def _capture() -> np.ndarray:
            recording = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="float32")  # type: ignore[attr-defined]
            sd.wait()
            return recording.reshape(-1)

        try:
            data = await asyncio.to_thread(_capture)
        except Exception as exc:  # pragma: no cover - hardware dependent
            raise AudioDeviceError(str(exc)) from exc
        return np.asarray(data, dtype=np.float32)


class WakeAcknowledgement:
    """Play a short acknowledgement when the wake word fires."""

    def __init__(self, sound_path: Optional[Path] = None) -> None:
        default_path = Path(__file__).resolve().parents[1] / "assets" / "rex_wake_acknowledgment (1).wav"
        self._sound_path = Path(sound_path) if sound_path else default_path

    async def play(self) -> None:
        if sa is None or not self._sound_path.exists():
            return

        def _play() -> None:
            wave_obj = sa.WaveObject.from_wave_file(str(self._sound_path))  # type: ignore[attr-defined]
            play_obj = wave_obj.play()
            play_obj.wait_done()

        try:
            await asyncio.to_thread(_play)
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.warning("Failed to play wake acknowledgement: %s", exc)


class SpeechToText:
    """Wrapper around Whisper transcription with asyncio integration."""

    def __init__(self, model_name: str, device: str) -> None:
        if whisper is None:
            raise SpeechToTextError("openai-whisper is not installed")
        try:
            self._model = whisper.load_model(model_name, device=device)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - dependency specific
            raise SpeechToTextError(str(exc)) from exc

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        def _transcribe() -> str:
            result = self._model.transcribe(audio, language="en", fp16=False)
            return str(result.get("text", "")).strip()

        try:
            text = await asyncio.to_thread(_transcribe)
        except Exception as exc:  # pragma: no cover - dependency specific
            raise SpeechToTextError(str(exc)) from exc
        return text


class TextToSpeech:
    """Text-to-speech helper that plays audio via simpleaudio when available."""

    def __init__(self, *, language: str, default_speaker: Optional[str] = None) -> None:
        self._language = language
        self._default_speaker = default_speaker if default_speaker and Path(default_speaker).exists() else None
        self._tts = None
        if TTS is not None:
            try:
                self._tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - dependency specific
                logger.warning("Unable to initialise TTS backend: %s", exc)
                self._tts = None

    async def speak(self, text: str, *, speaker_wav: Optional[str] = None) -> None:
        if not text:
            return

        if self._tts is None:
            logger.info("TTS disabled; response: %s", text)
            return

        speaker = speaker_wav or self._default_speaker

        def _synthesise() -> None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as handle:
                temp_path = Path(handle.name)
            try:
                self._tts.tts_to_file(  # type: ignore[attr-defined]
                    text=text,
                    speaker_wav=speaker,
                    language=self._language,
                    file_path=str(temp_path),
                )
                if sa is not None:
                    wave_obj = sa.WaveObject.from_wave_file(str(temp_path))  # type: ignore[attr-defined]
                    play_obj = wave_obj.play()
                    play_obj.wait_done()
                else:
                    logger.info("Synthesised speech stored at %s", temp_path)
            finally:
                with suppress(FileNotFoundError):
                    temp_path.unlink()

        try:
            await asyncio.to_thread(_synthesise)
        except Exception as exc:  # pragma: no cover - dependency specific
            raise TextToSpeechError(str(exc)) from exc


FrameSource = Callable[[], Awaitable[np.ndarray]]
PhraseRecorder = Callable[[], Awaitable[np.ndarray]]
Transcriber = Callable[[np.ndarray], Awaitable[str]]
Speaker = Callable[[str], Awaitable[None]]
Acknowledgement = Callable[[], Awaitable[None]]


class VoiceLoop:
    """Coordinate the complete voice interaction pipeline."""

    def __init__(
        self,
        assistant: Assistant,
        *,
        wake_listener: WakeWordListener,
        detection_source: FrameSource,
        record_phrase: PhraseRecorder,
        transcribe: Transcriber,
        speak: Speaker,
        acknowledge: Optional[Acknowledgement] = None,
    ) -> None:
        self._assistant = assistant
        self._wake_listener = wake_listener
        self._detection_source = detection_source
        self._record_phrase = record_phrase
        self._transcribe = transcribe
        self._speak = speak
        self._acknowledge = acknowledge

    async def run(self, *, max_interactions: Optional[int] = None) -> None:
        interactions = 0
        while max_interactions is None or interactions < max_interactions:
            try:
                await self._await_wakeword()
            except AudioDeviceError as exc:
                logger.error("Audio error while listening for wake word: %s", exc)
                interactions += 1
                await asyncio.sleep(1)
                continue
            except WakeWordError as exc:
                logger.error("Wake-word listener failed: %s", exc)
                interactions += 1
                await asyncio.sleep(1)
                continue

            interactions += 1

            if self._acknowledge:
                try:
                    await self._acknowledge()
                except Exception as exc:  # pragma: no cover - acknowledgement is best effort
                    logger.warning("Wake acknowledgement failed: %s", exc)

            try:
                audio = await self._record_phrase()
            except AudioDeviceError as exc:
                logger.error("Failed to record microphone input: %s", exc)
                continue

            try:
                transcript = await self._transcribe(audio)
            except SpeechToTextError as exc:
                logger.error("Transcription failed: %s", exc)
                continue

            if not transcript.strip():
                logger.info("Ignoring empty transcription")
                continue

            try:
                response = await self._assistant.generate_reply(transcript)
            except Exception as exc:  # pragma: no cover - generation should rarely fail
                logger.exception("Assistant failed to generate reply: %s", exc)
                continue

            try:
                await self._speak(response)
            except TextToSpeechError as exc:
                logger.error("Text-to-speech failed: %s", exc)

    async def _await_wakeword(self) -> None:
        try:
            async for _frame in self._wake_listener.listen(self._detection_source):
                self._wake_listener.stop()
                return
        except AudioDeviceError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            raise WakeWordError(str(exc)) from exc

        raise WakeWordError("Wake-word listener exited unexpectedly")


def _resolve_voice_reference() -> Optional[str]:
    try:
        users_map = load_users_map()
        profiles = load_all_profiles()
        user_key = resolve_user_key(settings.user_id, users_map, profiles=profiles)
        if user_key and user_key in profiles:
            reference = extract_voice_reference(profiles[user_key])
            if reference and Path(reference).exists():
                return reference
    except Exception:  # pragma: no cover - filesystem best effort
        logger.debug("Unable to resolve voice reference", exc_info=True)
    return None


def build_voice_loop(assistant: Assistant) -> VoiceLoop:
    if sd is not None and (settings.input_device is not None or settings.output_device is not None):
        sd.default.device = (settings.input_device, settings.output_device)  # type: ignore[attr-defined]

    microphone = AsyncMicrophone(
        sample_rate=settings.sample_rate,
        detection_seconds=settings.detection_frame_seconds,
        capture_seconds=settings.capture_seconds,
    )

    try:
        wake_model, _ = load_wakeword_model(keyword=settings.wakeword_keyword)
    except Exception as exc:
        raise WakeWordError(str(exc)) from exc

    detector = build_default_detector(wake_model, threshold=settings.wakeword_threshold)
    wake_listener = WakeWordListener(detector, poll_interval=settings.wakeword_poll_interval)
    acknowledgement = WakeAcknowledgement()

    speech_to_text: Optional[SpeechToText]
    try:
        speech_to_text = SpeechToText(settings.whisper_model, settings.whisper_device)
    except SpeechToTextError as exc:
        logger.warning("Speech-to-text unavailable: %s", exc)
        speech_to_text = None

    voice_reference = _resolve_voice_reference()
    text_to_speech = TextToSpeech(language=settings.speak_language, default_speaker=voice_reference)

    async def detection_source() -> np.ndarray:
        return await microphone.detection_frame()

    async def record_phrase() -> np.ndarray:
        return await microphone.record_phrase()

    async def transcribe(audio: np.ndarray) -> str:
        if speech_to_text is None:
            raise SpeechToTextError("Speech-to-text backend unavailable")
        return await speech_to_text.transcribe(audio, microphone.sample_rate)

    async def speak(text: str) -> None:
        await text_to_speech.speak(text, speaker_wav=voice_reference)

    async def acknowledge() -> None:
        await acknowledgement.play()

    return VoiceLoop(
        assistant,
        wake_listener=wake_listener,
        detection_source=detection_source,
        record_phrase=record_phrase,
        transcribe=transcribe,
        speak=speak,
        acknowledge=acknowledge,
    )


__all__ = [
    "AsyncMicrophone",
    "WakeAcknowledgement",
    "SpeechToText",
    "TextToSpeech",
    "VoiceLoop",
    "build_voice_loop",
]
