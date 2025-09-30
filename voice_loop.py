"""Async event loop orchestrating Rex's wake-word, STT, LLM and TTS pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

try:  # pragma: no cover - optional runtime dependencies
    import numpy as np
except ImportError:  # pragma: no cover - exercised when numpy is absent
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional runtime dependencies
    import sounddevice as sd
except ImportError:  # pragma: no cover - exercised when sounddevice is absent
    sd = None  # type: ignore[assignment]

try:  # pragma: no cover - optional runtime dependencies
    import simpleaudio as sa
except ImportError:  # pragma: no cover - exercised when simpleaudio is absent
    sa = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from TTS.api import TTS
    from faster_whisper import WhisperModel
else:  # pragma: no cover - runtime fallbacks when dependencies missing
    TTS = Any  # type: ignore[misc, assignment]
    WhisperModel = Any  # type: ignore[misc, assignment]

try:  # pragma: no cover - optional dependency for auto-detection
    import pyaudio
except ImportError:
    pyaudio = None

from assistant_errors import (
    SpeechRecognitionError,
    TextToSpeechError,
    WakeWordError,
)
from config import AppConfig, load_config
from logging_utils import get_logger
from llm_client import LanguageModel
try:  # pragma: no cover - exercised when persistence backends are optional
    from memory_utils import (
        append_history_entry,
        export_transcript,
        load_all_profiles,
        load_users_map,
        resolve_user_key,
        extract_voice_reference,
    )
except ImportError:  # pragma: no cover - fallback when TinyDB or deps unavailable
    _MEMORY: dict[str, list[dict[str, Any]]] = {}
    _PROFILES: dict[str, dict[str, Any]] = {}

    def append_history_entry(user: str, entry: dict[str, Any]) -> None:
        _MEMORY.setdefault(user, []).append(entry)

    def export_transcript(user: str, transcript: list[dict[str, Any]]) -> None:
        _MEMORY[user] = transcript

    def load_all_profiles() -> dict[str, dict[str, Any]]:
        return _PROFILES

    def load_users_map() -> dict[str, str]:
        return {user: user for user in _MEMORY}

    def resolve_user_key(candidate: Optional[str], users_map: dict[str, str], profiles=None):
        if candidate and candidate in users_map:
            return candidate
        return candidate

    def extract_voice_reference(profile: dict[str, Any]) -> Optional[str]:
        if isinstance(profile, dict):
            value = profile.get("voice_reference")
            return str(value) if value is not None else None
        return None

from plugin_loader import load_plugins
from wakeword_utils import detect_wakeword, load_wakeword_model

LOGGER = get_logger(__name__)


@dataclass
class WakeWordListener:
    """Bridge between the sounddevice callback and asyncio."""

    model: object
    threshold: float
    sample_rate: int
    block_size: int
    device: Optional[int]
    loop: asyncio.AbstractEventLoop

    def __post_init__(self) -> None:
        self._event = asyncio.Event()
        self._stream: Optional[object] = None

    def start(self) -> None:
        if sd is None:
            raise WakeWordError("sounddevice is required for wake-word detection")
        try:
            self._stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                callback=self._callback,
                device=self.device,
            )
            self._stream.start()
            LOGGER.info("Wake-word listener started")
        except Exception as exc:
            raise WakeWordError(f"Failed to start input stream: {exc}")

    def stop(self) -> None:
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
                self._stream.close()
            LOGGER.info("Wake-word listener stopped")
        self._event.set()

    def trigger(self) -> None:
        self.loop.call_soon_threadsafe(self._event.set)

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            LOGGER.warning("Audio status: %s", status)
        if np is None:
            raise WakeWordError("numpy is required for wake-word detection")
        audio_data = np.squeeze(indata)
        try:
            if detect_wakeword(self.model, audio_data, threshold=self.threshold):
                self.loop.call_soon_threadsafe(self._event.set)
        except Exception as exc:
            LOGGER.error("Wake-word detection failed: %s", exc)

    async def wait_for_wake(self) -> None:
        await self._event.wait()
        self._event.clear()


class AsyncRexAssistant:
    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or load_config()
        self.loop = asyncio.get_event_loop()
        self.language_model = LanguageModel(self.config)
        self.input_device, self.output_device = self._resolve_audio_devices()
        self._wake_model, self._wake_keyword = load_wakeword_model(
            keyword=self.config.wakeword, sensitivity=self.config.wakeword_sensitivity
        )
        frames_per_block = max(1, int(self.config.wakeword_window * self._wake_model.sample_rate / self._wake_model.frame_length))
        block_size = self._wake_model.frame_length * frames_per_block
        self._listener = WakeWordListener(
            model=self._wake_model,
            threshold=self.config.wakeword_threshold,
            sample_rate=self._wake_model.sample_rate,
            block_size=block_size,
            device=self.input_device if isinstance(self.input_device, int) else None,
            loop=self.loop,
        )
        self._tts: Optional[Any] = None
        self._whisper_model: Optional[Any] = None
        self.users_map = load_users_map()
        self.profiles = load_all_profiles()
        resolved = resolve_user_key(self.config.default_user, self.users_map, profiles=self.profiles)
        self.active_user = resolved or (self.config.default_user or "james")
        active_profile = self.profiles.get(self.active_user, {}) if isinstance(self.profiles, dict) else {}
        self.voice_reference = extract_voice_reference(active_profile) if active_profile else None
        self.plugins = load_plugins()
        LOGGER.info("Loaded plugins: %s", ", ".join(self.plugins.keys()) or "none")
        self._running = True

    def _auto_detect_device(self, kind: str) -> Optional[int]:
        if pyaudio is None:  # pragma: no cover - dependency optional
            return None
        pa = pyaudio.PyAudio()
        try:
            if kind == "input":
                info = pa.get_default_input_device_info()
            else:
                info = pa.get_default_output_device_info()
            if not info:
                return None
            index = info.get("index")
            return int(index) if isinstance(index, int) else None
        except OSError:
            return None
        finally:
            pa.terminate()

    def _resolve_audio_devices(self) -> tuple[Optional[int], Optional[int]]:
        def normalize(value: object) -> Optional[int]:
            if value in (None, "default"):
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
            return None

        input_device = normalize(self.config.audio_input_device)
        output_device = normalize(self.config.audio_output_device)

        if input_device is None:
            detected = self._auto_detect_device("input")
            if detected is not None:
                input_device = detected

        if output_device is None:
            detected = self._auto_detect_device("output")
            if detected is not None:
                output_device = detected

        if sd is not None:
            sd.default.device = (
                input_device if isinstance(input_device, int) else None,
                output_device if isinstance(output_device, int) else None,
            )

        return input_device, output_device

    def _get_tts(self) -> TTS:
        if self._tts is None:
            try:
                from TTS.api import TTS as CoquiTTS
            except ImportError as exc:  # pragma: no cover - exercised when dependency missing
                raise TextToSpeechError("Coqui TTS library is required for speech synthesis") from exc

            gpu_enabled = bool(self.config.gpu)
            if gpu_enabled:
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.config.cuda_device))
                try:
                    import torch

                    if not torch.cuda.is_available():
                        LOGGER.warning("CUDA requested for TTS but no GPU is available; falling back to CPU.")
                        gpu_enabled = False
                except ImportError:
                    LOGGER.warning("PyTorch not available; falling back to CPU-only TTS.")
                    gpu_enabled = False
            self._tts = CoquiTTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
                gpu=gpu_enabled,
            )
        return self._tts

    def _get_whisper(self) -> Any:
        if self._whisper_model is None:
            try:
                from faster_whisper import WhisperModel as FasterWhisperModel
            except ImportError as exc:  # pragma: no cover - exercised when dependency missing
                raise SpeechRecognitionError("faster-whisper is required for transcription") from exc

            try:
                preferred = (str(self.config.whisper_device).lower() if self.config.whisper_device else "auto")
                prefer_gpu = preferred in {"cuda", "gpu"} or (preferred == "auto" and self.config.gpu)
                device = "cuda" if prefer_gpu else "cpu"
                if device == "cuda":
                    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.config.cuda_device))
                    try:
                        import torch

                        if not torch.cuda.is_available():
                            LOGGER.warning("CUDA requested for Whisper but not available; using CPU fallback.")
                            device = "cpu"
                    except ImportError:
                        LOGGER.warning("PyTorch not available; using CPU fallback for Whisper.")
                        device = "cpu"
                compute_type = (
                    self.config.whisper_compute_type if device == "cuda" else "int8"
                )
                self._whisper_model = FasterWhisperModel(
                    self.config.whisper_model,
                    device=device,
                    compute_type=compute_type,
                )
            except Exception as exc:
                raise SpeechRecognitionError(f"Failed to load Whisper model: {exc}")
        return self._whisper_model

    async def run(self) -> None:
        self._listener.start()
        try:
            while self._running:
                await self._listener.wait_for_wake()
                if not self._running:
                    break
                LOGGER.info("Wake word '%s' detected", self._wake_keyword)
                await self._handle_interaction()
        finally:
            self._listener.stop()

    async def _handle_interaction(self) -> None:
        await asyncio.gather(
            asyncio.to_thread(self._play_wake_sound),
            self._process_conversation(),
        )

    async def _process_conversation(self) -> None:
        audio = await asyncio.to_thread(self._record_audio)
        try:
            transcript = await asyncio.to_thread(self._transcribe_audio, audio)
        except SpeechRecognitionError as exc:
            LOGGER.error("Transcription failed: %s", exc)
            return

        if not transcript:
            LOGGER.info("No speech detected after wake word")
            return

        if self._maybe_switch_user(transcript):
            LOGGER.info("Switched active user to %s", self.active_user)
            return

        LOGGER.info("User (%s): %s", self.active_user, transcript)
        append_history_entry(
            self.active_user,
            {"role": "user", "text": transcript},
        )

        response = await asyncio.to_thread(self.language_model.generate, transcript)
        append_history_entry(
            self.active_user,
            {"role": "assistant", "text": response},
        )

        if self.config.conversation_export:
            export_transcript(
                self.active_user,
                [
                    {"role": "user", "text": transcript},
                    {"role": "assistant", "text": response},
                ],
            )

        try:
            await asyncio.to_thread(self._speak_response, response)
        except TextToSpeechError as exc:
            LOGGER.error("TTS failed: %s", exc)

    def _record_audio(self) -> Any:
        if np is None:
            raise SpeechRecognitionError("numpy is required for audio capture")
        if sd is None:
            raise SpeechRecognitionError("sounddevice is required for audio capture")
        sample_rate = self._wake_model.sample_rate
        duration = self.config.command_duration
        frames = int(sample_rate * duration)
        LOGGER.info("Recording %.1f seconds of audio", duration)

        buffer: list[np.ndarray] = []
        collected = 0
        finished = threading.Event()

        def callback(indata, _frames, _time, status) -> None:
            nonlocal collected
            if status:
                LOGGER.warning("Recorder status: %s", status)
            buffer.append(indata.copy())
            collected += len(indata)
            if collected >= frames:
                finished.set()
                raise sd.CallbackStop()

        device = self.input_device if isinstance(self.input_device, int) else None
        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                blocksize=self._wake_model.frame_length,
                dtype="float32",
                device=device,
                callback=callback,
            ):
                finished.wait(timeout=duration + 1)
        except sd.CallbackStop:
            pass
        except Exception as exc:
            raise SpeechRecognitionError(f"Failed to capture microphone audio: {exc}") from exc

        if not buffer:
            return np.zeros(frames, dtype=np.float32)

        audio = np.concatenate(buffer, axis=0)[:frames]
        return np.squeeze(audio).astype(np.float32)

    def _transcribe_audio(self, audio: Any) -> str:
        model = self._get_whisper()
        try:
            segments, _info = model.transcribe(
                audio.astype(np.float32),
                beam_size=5,
                vad_filter=True,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            raise SpeechRecognitionError(str(exc))
        return text

    def _speak_response(self, text: str) -> None:
        if np is None:
            raise TextToSpeechError("numpy is required for speech playback")
        if sd is None:
            raise TextToSpeechError("sounddevice is required for speech playback")
        tts = self._get_tts()
        if not text.strip():
            return
        try:
            waveform = tts.tts(
                text=text,
                speaker_wav=self.voice_reference,
                language=self.config.speak_language,
            )
            waveform = np.array(waveform, dtype=np.float32)
            device = self.output_device if isinstance(self.output_device, int) else None
            sd.play(waveform, samplerate=tts.synthesizer.output_sample_rate, device=device)
            sd.wait()
        except Exception as exc:
            raise TextToSpeechError(f"Failed to synthesise speech: {exc}")

    def _play_wake_sound(self) -> None:
        path = self.config.wake_sound_path
        if not path:
            return
        if sa is None:
            LOGGER.warning("simpleaudio is unavailable; skipping wake sound")
            return
        try:
            wave_obj = sa.WaveObject.from_wave_file(path)
            play_obj = wave_obj.play()
            play_obj.wait_done()
        except Exception as exc:
            LOGGER.warning("Failed to play wake sound %s: %s", path, exc)

    def _maybe_switch_user(self, transcript: str) -> bool:
        cleaned = transcript.strip().lower()
        if not cleaned.startswith("this is "):
            return False
        candidate = cleaned.replace("this is ", "", 1).strip().strip(".")
        resolved = resolve_user_key(candidate, self.users_map, profiles=self.profiles)
        if resolved:
            self.active_user = resolved
            return True
        return False

    def stop(self) -> None:
        self._running = False
        self._listener.trigger()


def build_voice_loop(assistant: Optional[AsyncRexAssistant] = None) -> AsyncRexAssistant:
    """Helper function for rex_loop.py to build and run the voice assistant loop."""
    return assistant or AsyncRexAssistant()
