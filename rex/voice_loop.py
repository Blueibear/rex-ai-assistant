"""Async voice assistant loop orchestrating wake word, STT, LLM, and TTS."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from collections.abc import Awaitable
from contextlib import suppress
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.io import wavfile
import soundfile as sf

from .assistant import Assistant
from .assistant_errors import AudioDeviceError, SpeechToTextError, TextToSpeechError, WakeWordError
from .config import settings
from .memory import (
    extract_voice_reference,
    load_all_profiles,
    load_users_map,
    resolve_user_key,
)
from .wakeword.listener import WakeWordListener, build_default_detector
from .wakeword.utils import load_wakeword_model

try:
    import simpleaudio as sa  # type: ignore
except ImportError:
    sa = None

try:
    import sounddevice as sd  # type: ignore
except ImportError:
    sd = None

try:
    from TTS.api import TTS  # type: ignore
except ImportError:
    TTS = None

try:
    import whisper  # type: ignore
except ImportError:
    whisper = None

logger = logging.getLogger(__name__)

RecorderCallable = Callable[[float], Awaitable[np.ndarray] | np.ndarray]


class AsyncMicrophone:
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

    async def record_phrase(self, duration: float | None = None) -> np.ndarray:
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
        except Exception as exc:
            raise AudioDeviceError(str(exc)) from exc
        return np.asarray(data, dtype=np.float32)


class WakeAcknowledgement:
    def __init__(self, sound_path: Path | None = None) -> None:
        default_path = Path(__file__).resolve().parents[1] / "assets" / "wake_acknowledgment.wav"
        self._sound_path = sound_path or default_path

    async def play(self) -> None:
        logger.debug(f"[WAKE ACK] Attempting to play: {self._sound_path}")
        if sa is None:
            logger.warning("[WAKE ACK] simpleaudio not available, skipping acknowledgment")
            return

        if not self._sound_path.exists():
            logger.warning(f"[WAKE ACK] Sound file not found: {self._sound_path}")
            return

        def _play() -> None:
            wave_obj = sa.WaveObject.from_wave_file(str(self._sound_path))
            play_obj = wave_obj.play()
            play_obj.wait_done()

        try:
            await asyncio.to_thread(_play)
            logger.debug("[WAKE ACK] Successfully played acknowledgment")
        except Exception as exc:
            logger.warning("Failed to play wake acknowledgement: %s", exc)


class SpeechToText:
    def __init__(self, model_name: str, device: str) -> None:
        if whisper is None:
            raise SpeechToTextError("openai-whisper is not installed")
        try:
            self._model = whisper.load_model(model_name, device=device)
        except Exception as exc:
            raise SpeechToTextError(str(exc)) from exc

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        def _transcribe() -> str:
            logger.debug(f"[STT] Transcribing audio with shape={audio.shape}, dtype={audio.dtype}")
            result = self._model.transcribe(audio, language="en", fp16=False)
            logger.debug(f"[STT] Raw Whisper result: {result}")
            return str(result.get("text", "")).strip()

        try:
            return await asyncio.to_thread(_transcribe)
        except Exception as exc:
            logger.exception("[STT] Whisper failed with exception:")
            raise SpeechToTextError(str(exc)) from exc


class TextToSpeech:
    def __init__(self, *, language: str, default_speaker: str | None = None) -> None:
        self._language = language
        self._default_speaker = (
            default_speaker if default_speaker and Path(default_speaker).exists() else None
        )
        self._provider = os.getenv("REX_TTS_PROVIDER", "xtts").lower()
        self._edge_voice = os.getenv("REX_TTS_VOICE", "en-US-AndrewNeural")
        self._piper_model = Path(os.getenv("REX_PIPER_MODEL", "voices/en_US-lessac-medium.onnx"))

        self._tts = None
        if self._provider == "xtts" and TTS is not None:
            try:
                import torch
                self._tts = TTS(
                    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                    progress_bar=False,
                )
                if torch.cuda.is_available():
                    self._tts.to("cuda")
                    logger.info("[TTS] Loaded XTTS v2 on CUDA")
                else:
                    logger.info("[TTS] Loaded XTTS v2 on CPU")
            except Exception as exc:
                logger.warning("Unable to initialise XTTS: %s", exc)
        elif self._provider == "edge":
            logger.info("[TTS] Using Edge-TTS")
        elif self._provider == "piper":
            logger.info("[TTS] Using Piper TTS")
        elif self._provider == "windows":
            logger.info("[TTS] Using Windows SAPI")

    async def speak(self, text: str, *, speaker_wav: str | None = None) -> None:
        if not text:
            return

        text = self._clean_text(text)
        if not text:
            logger.warning("[TTS] Nothing to speak after cleaning.")
            return

        logger.info("[TTS] Speaking: %s", text)

        try:
            if self._provider == "xtts":
                await self._speak_xtts(text, speaker_wav)
            elif self._provider == "edge":
                await self._speak_edge(text)
            elif self._provider == "piper":
                await self._speak_piper(text)
            elif self._provider == "windows":
                await self._speak_windows(text)
            else:
                logger.warning("Unknown TTS provider: %s", self._provider)
                print(f"Rex: {text}")
        except Exception as exc:
            logger.error("[TTS] Failed: %s", exc)
            print(f"Rex: {text}")

    def _clean_text(self, text: str) -> str:
        if "Additional info:" in text:
            text = text.split("Additional info:")[0].strip()
        text = re.sub(r"http[s]?://\S+", "", text)
        text = re.sub(r"uddg=\S+", "", text)
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        text = ". ".join(sentences[:2])
        if text and not text.endswith("."):
            text += "."
        return text

    async def _speak_xtts(self, text: str, speaker_wav: str | None) -> None:
        if self._tts is None:
            raise TextToSpeechError("XTTS not initialized")

        speaker = speaker_wav or self._default_speaker

        def _synthesise() -> None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as handle:
                temp_path = Path(handle.name)
            try:
                kwargs: dict[str, object] = {
                    "text": text,
                    "language": self._language,
                    "file_path": str(temp_path),
                }
                if speaker:
                    kwargs["speaker_wav"] = speaker
                else:
                    kwargs["speaker"] = "Claribel Dervla"

                self._tts.tts_to_file(**kwargs)
                if sa is not None:
                    wave_obj = sa.WaveObject.from_wave_file(str(temp_path))
                    play_obj = wave_obj.play()
                    play_obj.wait_done()
            finally:
                with suppress(FileNotFoundError):
                    temp_path.unlink()

        await asyncio.to_thread(_synthesise)

    # _speak_edge, _speak_piper, and _speak_windows remain unchanged

# The rest of the file (VoiceLoop, _resolve_voice_reference, build_voice_loop) remains unchanged.
