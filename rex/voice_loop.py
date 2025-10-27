"""Async voice assistant loop orchestrating wake word, STT, LLM, and TTS."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

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


@dataclass
class SynthesizedAudio:
    """Container for synthesized audio suitable for playback or transport."""

    sample_rate: int
    pcm: np.ndarray  # Float32 mono samples in range [-1, 1]

    def to_int16(self) -> np.ndarray:
        """Return the PCM data as signed 16-bit integers."""
        return (np.clip(self.pcm, -1.0, 1.0) * 32767.0).astype(np.int16)


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
        default_path = Path(__file__).resolve().parents[1] / "assets" / "wake_acknowledgment.wav"
        self._sound_path = Path(sound_path) if sound_path else default_path

    async def play(self) -> None:
        logger.debug(f"[WAKE ACK] Attempting to play: {self._sound_path}")
        logger.debug(f"[WAKE ACK] File exists: {self._sound_path.exists()}")
        logger.debug(f"[WAKE ACK] simpleaudio available: {sa is not None}")
        
        if sa is None:
            logger.warning("[WAKE ACK] simpleaudio not available, skipping acknowledgment")
            return
        
        if not self._sound_path.exists():
            logger.warning(f"[WAKE ACK] Sound file not found: {self._sound_path}")
            return

        def _play() -> None:
            wave_obj = sa.WaveObject.from_wave_file(str(self._sound_path))  # type: ignore[attr-defined]
            play_obj = wave_obj.play()
            play_obj.wait_done()

        try:
            await asyncio.to_thread(_play)
            logger.debug("[WAKE ACK] Successfully played acknowledgment")
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
    """Configurable TTS with multiple backends."""

    def __init__(self, *, language: str, default_speaker: Optional[str] = None) -> None:
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
                # Move to CUDA if available
                if torch.cuda.is_available():
                    self._tts.to("cuda")
                    logger.info("[TTS] Loaded XTTS v2 on CUDA (local, high quality)")
                else:
                    logger.info("[TTS] Loaded XTTS v2 on CPU (local, high quality, slow)")
            except Exception as exc:  # pragma: no cover - dependency specific
                logger.warning("Unable to initialise XTTS: %s", exc)
        elif self._provider == "edge":
            logger.info("[TTS] Using Edge-TTS (cloud, fast, high quality)")
        elif self._provider == "piper":
            logger.info("[TTS] Using Piper (local, fast, good quality)")
        elif self._provider == "windows":
            logger.info("[TTS] Using Windows SAPI (local, instant, robotic)")

    async def speak(self, text: str, *, speaker_wav: Optional[str] = None) -> None:
        if not text:
            return

        text = self._clean_text(text)
        if not text:
            logger.warning("[TTS] Nothing to speak after cleaning.")
            return

        logger.info("[TTS] Speaking: %s", text)

        try:
            if self._provider == "xtts":
                audio = await self._synthesise_xtts_audio(text, speaker_wav)
                if sa is None:
                    logger.warning("[TTS] simpleaudio not available, printing response.")
                    print(f"Rex: {text}")
                    return

                def _play() -> None:
                    samples = audio.to_int16()
                    play_obj = sa.play_buffer(samples.tobytes(), 1, 2, audio.sample_rate)  # type: ignore[attr-defined]
                    play_obj.wait_done()

                await asyncio.to_thread(_play)
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

    async def synthesise(self, text: str, *, speaker_wav: Optional[str] = None) -> SynthesizedAudio:
        """Generate speech audio without playing it."""
        if not text.strip():
            raise TextToSpeechError("Text must not be empty for synthesis")

        cleaned = self._clean_text(text)
        if not cleaned:
            raise TextToSpeechError("Nothing to synthesise after cleaning")

        if self._provider != "xtts":
            raise TextToSpeechError(
                f"Audio capture not supported for provider '{self._provider}'"
            )

        return await self._synthesise_xtts_audio(cleaned, speaker_wav)

    def _clean_text(self, text: str) -> str:
        if "Additional info:" in text:
            text = text.split("Additional info:")[0].strip()

        text = re.sub(r"http[s]?://\S+", "", text)
        text = re.sub(r"uddg=\S+", "", text)

        sentences = [sentence.strip() for sentence in text.split(".") if sentence.strip()]
        text = ". ".join(sentences[:2])
        if text and not text.endswith("."):
            text += "."
        return text

    async def _synthesise_xtts_audio(
        self, text: str, speaker_wav: Optional[str]
    ) -> SynthesizedAudio:
        if self._tts is None:
            raise TextToSpeechError("XTTS not initialized")

        speaker = speaker_wav or self._default_speaker

        def _render() -> SynthesizedAudio:
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

                self._tts.tts_to_file(**kwargs)  # type: ignore[attr-defined]
                data, sample_rate = sf.read(temp_path, dtype="float32")
                if data.ndim > 1:
                    data = data[:, 0]
                return SynthesizedAudio(sample_rate=sample_rate, pcm=np.asarray(data, dtype=np.float32))
            finally:
                with suppress(FileNotFoundError):
                    temp_path.unlink()

        return await asyncio.to_thread(_render)

    async def _speak_edge(self, text: str) -> None:
        try:
            import edge_tts
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise TextToSpeechError(
                "edge-tts not installed. Run: pip install edge-tts"
            ) from exc

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as handle:
            temp_path = Path(handle.name)
        try:
            communicate = edge_tts.Communicate(text, self._edge_voice)
            await communicate.save(str(temp_path))

            if sa is not None:
                import subprocess

                wav_path = temp_path.with_suffix(".wav")
                subprocess.run(
                    ["ffmpeg", "-i", str(temp_path), "-y", str(wav_path)],
                    capture_output=True,
                    check=True,
                )

                wave_obj = sa.WaveObject.from_wave_file(str(wav_path))  # type: ignore[attr-defined]
                play_obj = wave_obj.play()
                play_obj.wait_done()
                wav_path.unlink()
        finally:
            with suppress(FileNotFoundError):
                temp_path.unlink()

    async def _speak_piper(self, text: str) -> None:
        if not self._piper_model.exists():
            raise TextToSpeechError(f"Piper model not found: {self._piper_model}")

        def _synthesise() -> None:
            try:
                from piper import PiperVoice  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise TextToSpeechError(
                    "piper-tts not installed. Run: pip install piper-tts"
                ) from exc

            import wave

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as handle:
                temp_path = Path(handle.name)
            try:
                voice = PiperVoice.load(str(self._piper_model))
                with wave.open(str(temp_path), "wb") as wav_file:
                    voice.synthesize(text, wav_file)

                if sa is not None:
                    wave_obj = sa.WaveObject.from_wave_file(str(temp_path))  # type: ignore[attr-defined]
                    play_obj = wave_obj.play()
                    play_obj.wait_done()
            finally:
                with suppress(FileNotFoundError):
                    temp_path.unlink()

        await asyncio.to_thread(_synthesise)

    async def _speak_windows(self, text: str) -> None:
        def _speak() -> None:
            try:
                import pyttsx3
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise TextToSpeechError(
                    "pyttsx3 not installed. Run: pip install pyttsx3"
                ) from exc

            engine = pyttsx3.init()
            
            # Allow voice selection via environment variable
            voice_index = os.getenv("REX_WINDOWS_TTS_VOICE_INDEX")
            if voice_index is not None:
                try:
                    voices = engine.getProperty('voices')
                    engine.setProperty('voice', voices[int(voice_index)].id)
                except (IndexError, ValueError):
                    logger.warning(f"Invalid voice index: {voice_index}, using default")
            
            engine.setProperty("rate", 180)
            engine.setProperty("volume", 0.9)
            engine.say(text)
            engine.runAndWait()

        await asyncio.to_thread(_speak)


FrameSource = Callable[[], Awaitable[np.ndarray]]
PhraseRecorder = Callable[[], Awaitable[np.ndarray]]
Transcriber = Callable[[np.ndarray], Awaitable[str]]
Speaker = Callable[[str], Awaitable[None]]
Acknowledgement = Callable[[], Awaitable[None]]


class VoiceLoop:
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
                    logger.debug("🔊 Playing wake acknowledgment...")
                    await self._acknowledge()
                except Exception as exc:
                    logger.warning("Wake acknowledgement failed: %s", exc)

            try:
                audio = await self._record_phrase()
                logger.debug("[MIC] Finished recording audio input.")
            except AudioDeviceError as exc:
                logger.error("Failed to record microphone input: %s", exc)
                continue

            try:
                transcript = await self._transcribe(audio)
                logger.info(f"[STT] Transcript: '{transcript}'")
            except Exception as exc:
                logger.error("[STT] Transcription failed: %s", exc)
                continue

            if not transcript.strip():
                logger.warning("[STT] Empty transcription, skipping...")
                continue

            try:
                response = await self._assistant.generate_reply(transcript)
                logger.info(f"[LLM] Assistant reply: '{response}'")
            except Exception as exc:
                logger.exception("Assistant failed to generate reply: %s", exc)
                continue

            if not response.strip():
                logger.warning("[LLM] Empty assistant response, skipping TTS...")
                continue

            if "Additional info:" in response:
                response = response.split("Additional info:")[0].strip()

            response = re.sub(r"http[s]?://\S+", "", response)
            response = re.sub(r"uddg=\S+", "", response)
            response = re.sub(r"\[.*?\]", "", response)

            sentences = [sentence.strip() for sentence in response.split(".") if sentence.strip()]
            response_to_speak = ". ".join(sentences[:2]).strip()
            if response_to_speak and not response_to_speak.endswith("."):
                response_to_speak += "."

            if not response_to_speak:
                logger.warning("[TTS] Cleaned response is empty, skipping speech synthesis...")
                continue

            logger.info(f"[TTS] Cleaned response: '{response_to_speak}'")

            try:
                logger.debug("[TTS] Speaking reply...")
                await self._speak(response_to_speak)
            except Exception as exc:
                logger.error("Text-to-speech failed: %s", exc)

    async def _await_wakeword(self) -> None:
        try:
            async for _frame in self._wake_listener.listen(self._detection_source):
                self._wake_listener.stop()
                return
        except AudioDeviceError:
            raise
        except Exception as exc:
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
    if sd is not None and (settings.audio_input_device is not None or settings.audio_output_device is not None):
        sd.default.device = (settings.audio_input_device, settings.audio_output_device)  # type: ignore[attr-defined]

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
        audio = await microphone.record_phrase()

        duration = len(audio) / microphone.sample_rate
        peak_volume = float(np.max(np.abs(audio)))
        logger.debug(
            "[MIC] Recorded phrase: %s samples, %.2fs, peak volume: %.4f",
            len(audio),
            duration,
            peak_volume,
        )

        debug_output = Path("debug_recording.wav")
        try:
            sf.write(debug_output, audio, microphone.sample_rate)
            logger.debug("[MIC] Saved debug audio to %s", debug_output)
        except Exception as exc:
            logger.warning("[MIC] Failed to save debug recording: %s", exc)

        print("[DEBUG] Finished recording phrase")
        try:
            wavfile.write("debug_recording.wav", microphone.sample_rate, (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16))
        except Exception as exc:
            logger.warning("[MIC] Failed to save debug recording (wavfile): %s", exc)

        return audio

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
    "SynthesizedAudio",
    "TextToSpeech",
    "VoiceLoop",
    "build_voice_loop",
]
