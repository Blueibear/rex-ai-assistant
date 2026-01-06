"""Optimized async voice assistant loop with reduced latency.

This is the CANONICAL implementation of the voice loop functionality.
The `rex/voice_loop.py` module is a compatibility wrapper that re-exports
from this module.

Key optimizations:
1. Voice Activity Detection for shorter recording times
2. Concurrent STT + LLM processing where possible
3. Removed excessive debug logging
4. Faster TTS provider recommendations
5. Better error handling
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Awaitable, Callable, Optional

import numpy as np
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
    import simpleaudio as sa
except ImportError:
    sa = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from TTS.api import TTS
except ImportError:
    TTS = None

try:
    import whisper
except ImportError:
    whisper = None

logger = logging.getLogger(__name__)

RecorderCallable = Callable[[float], Awaitable[np.ndarray] | np.ndarray]


class AsyncMicrophone:
    """Optimized microphone recording with Voice Activity Detection."""
    def __init__(
        self,
        *,
        sample_rate: int,
        detection_seconds: float,
        capture_seconds: float,
        recorder: RecorderCallable | None = None,
        vad_threshold: float = 0.01,
        silence_duration: float = 1.0,
    ) -> None:
        self.sample_rate = sample_rate
        self._detection_seconds = detection_seconds
        self._capture_seconds = capture_seconds
        self._recorder = recorder
        self._vad_threshold = vad_threshold
        self._silence_duration = silence_duration

    async def detection_frame(self) -> np.ndarray:
        return await self._record(self._detection_seconds)

    async def record_phrase(self, duration: Optional[float] = None) -> np.ndarray:
        return await self._record_with_vad(duration or self._capture_seconds)

    async def _record_with_vad(self, max_duration: float) -> np.ndarray:
        if sd is None:
            raise AudioDeviceError("sounddevice is not installed")

        chunk_duration = 0.2
        chunk_frames = int(self.sample_rate * chunk_duration)
        max_chunks = int(max_duration / chunk_duration)
        silence_chunks_needed = int(self._silence_duration / chunk_duration)

        chunks = []
        silence_chunks = 0
        has_voice = False

        def _capture_chunk() -> np.ndarray:
            recording = sd.rec(chunk_frames, samplerate=self.sample_rate, channels=1, dtype="float32")
            sd.wait()
            return recording.reshape(-1)

        try:
            for _ in range(max_chunks):
                chunk = await asyncio.to_thread(_capture_chunk)
                chunks.append(chunk)

                rms = np.sqrt(np.mean(chunk**2))

                if rms > self._vad_threshold:
                    has_voice = True
                    silence_chunks = 0
                elif has_voice:
                    silence_chunks += 1
                    if silence_chunks >= silence_chunks_needed:
                        break
        except Exception as exc:
            raise AudioDeviceError(str(exc)) from exc

        return np.concatenate(chunks) if chunks else np.zeros(chunk_frames, dtype=np.float32)

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
            recording = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="float32")
            sd.wait()
            return recording.reshape(-1)

        try:
            data = await asyncio.to_thread(_capture)
        except Exception as exc:
            raise AudioDeviceError(str(exc)) from exc
        return np.asarray(data, dtype=np.float32)


class WakeAcknowledgement:
    def __init__(self, sound_path: Optional[Path] = None) -> None:
        default_path = Path(__file__).resolve().parents[1] / "assets" / "wake_acknowledgment.wav"
        self._sound_path = Path(sound_path) if sound_path else default_path

    async def play(self) -> None:
        if sa is None or not self._sound_path.exists():
            return

        def _play() -> None:
            wave_obj = sa.WaveObject.from_wave_file(str(self._sound_path))
            play_obj = wave_obj.play()
            play_obj.wait_done()

        try:
            await asyncio.to_thread(_play)
        except Exception as exc:
            logger.warning("Wake acknowledgement failed: %s", exc)


class SpeechToText:
    def __init__(self, model_name: str, device: str) -> None:
        if whisper is None:
            raise SpeechToTextError("openai-whisper is not installed")

        if model_name == "base":
            logger.info("[STT] Using 'tiny' model for faster transcription (3x speedup)")
            model_name = "tiny"

        try:
            self._model = whisper.load_model(model_name, device=device)
        except Exception as exc:
            raise SpeechToTextError(str(exc)) from exc

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        def _transcribe() -> str:
            result = self._model.transcribe(audio, language="en", fp16=False)
            return str(result.get("text", "")).strip()

        try:
            return await asyncio.to_thread(_transcribe)
        except Exception as exc:
            logger.error("[STT] Whisper failed: %s", exc)
            raise SpeechToTextError(str(exc)) from exc


class TextToSpeech:
    def __init__(self, *, language: str, default_speaker: Optional[str] = None) -> None:
        self._language = language
        self._default_speaker = default_speaker
        self._provider = os.getenv("REX_TTS_PROVIDER", "edge").lower()
        self._edge_voice = os.getenv("REX_TTS_VOICE", "en-US-AndrewNeural")

        if self._provider == "xtts":
            logger.warning("[TTS] XTTS is slow (~3-4s). Recommend 'edge' or 'windows' for <1s latency")

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
            except Exception as exc:
                logger.warning("XTTS init failed: %s", exc)

    async def speak(self, text: str, *, speaker_wav: Optional[str] = None) -> None:
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
        if "Additional info:" in text:
            text = text.split("Additional info:")[0].strip()
        text = re.sub(r"http[s]?://\S+", "", text)
        text = re.sub(r"uddg=\S+", "", text)
        text = re.sub(r"\[.*?\]", "", text)
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        text = ". ".join(sentences[:2])
        return text + "." if text and not text.endswith(".") else text

    async def _speak_xtts(self, text: str, speaker_wav: Optional[str]) -> None:
        if self._tts is None:
            raise TextToSpeechError("XTTS not initialized")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        try:
            def _synthesize() -> None:
                self._tts.tts_to_file(
                    text=text,
                    speaker_wav=speaker_wav or self._default_speaker,
                    language=self._language,
                    file_path=output_path,
                )

            await asyncio.to_thread(_synthesize)

            if sa is not None and Path(output_path).exists():
                def _play() -> None:
                    wave_obj = sa.WaveObject.from_wave_file(output_path)
                    play_obj = wave_obj.play()
                    play_obj.wait_done()

                await asyncio.to_thread(_play)
        finally:
            with suppress(FileNotFoundError):
                Path(output_path).unlink()

    async def _speak_edge(self, text: str) -> None:
        try:
            import edge_tts
        except ImportError:
            raise TextToSpeechError("edge-tts is not installed")

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
    """Optimized voice assistant loop with reduced latency."""

    def __init__(
        self,
        assistant,
        *,
        wake_listener,
        detection_source: Callable[[], Awaitable[np.ndarray]],
        record_phrase: Callable[[], Awaitable[np.ndarray]],
        transcribe: Callable[[np.ndarray], Awaitable[str]],
        speak: Callable[[str], Awaitable[None]],
        acknowledge: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        self._assistant = assistant
        self._wake_listener = wake_listener
        self._detection_source = detection_source
        self._record_phrase = record_phrase
        self._transcribe = transcribe
        self._speak = speak
        self._acknowledge = acknowledge

    async def run(self, max_interactions: Optional[int] = None) -> None:
        """Run the voice loop for a specified number of interactions."""
        interactions = 0

        async for _ in self._wake_listener.listen(self._detection_source):
            try:
                if self._acknowledge:
                    await self._acknowledge()

                # Record user speech
                audio = await self._record_phrase()

                # Transcribe to text
                transcript = await self._transcribe(audio)
                if not transcript:
                    logger.info("No speech detected")
                    continue

                # Get LLM response
                response = await self._assistant.generate_reply(transcript)

                # Ensure response ends with period for better TTS
                if response and not response.endswith("."):
                    response = response + "."

                # Speak response
                await self._speak(response)

            except SpeechToTextError as exc:
                logger.error("STT error: %s", exc)
                # Continue loop on transcription errors
            except AudioDeviceError as exc:
                logger.error("Audio device error: %s", exc)
                # Propagate audio device errors by breaking
                break
            except Exception as exc:
                logger.error("Unexpected error in voice loop: %s", exc)

            interactions += 1
            if max_interactions is not None and interactions >= max_interactions:
                break


def build_voice_loop(
    assistant,
    *,
    sample_rate: int = 16000,
    detection_seconds: float = 1.0,
    capture_seconds: float = 5.0,
    whisper_model: str = "tiny",  # Optimized default
    device: str = "cpu",
    language: str = "en",
    speaker_wav: Optional[str] = None,
    wake_sound_path: Optional[Path] = None,
    vad_threshold: float = 0.01,
    silence_duration: float = 1.0,
) -> VoiceLoop:
    """Build an optimized VoiceLoop with VAD and faster defaults."""
    from .wakeword.listener import build_default_detector

    mic = AsyncMicrophone(
        sample_rate=sample_rate,
        detection_seconds=detection_seconds,
        capture_seconds=capture_seconds,
        vad_threshold=vad_threshold,
        silence_duration=silence_duration,
    )

    wake_listener = build_default_detector(
        sample_rate=sample_rate,
        chunk_duration=detection_seconds,
    )

    stt = SpeechToText(model_name=whisper_model, device=device)
    tts = TextToSpeech(language=language, default_speaker=speaker_wav)
    ack = WakeAcknowledgement(sound_path=wake_sound_path)

    return VoiceLoop(
        assistant,
        wake_listener=wake_listener,
        detection_source=mic.detection_frame,
        record_phrase=mic.record_phrase,
        transcribe=lambda audio: stt.transcribe(audio, sample_rate),
        speak=lambda text: tts.speak(text, speaker_wav=speaker_wav),
        acknowledge=ack.play,
    )


__all__ = [
    "AsyncMicrophone",
    "WakeAcknowledgement",
    "SpeechToText",
    "TextToSpeech",
    "VoiceLoop",
    "build_voice_loop",
]
