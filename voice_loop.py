"""Async event loop orchestrating Rex's wake-word, STT, LLM and TTS pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simpleaudio as sa
import sounddevice as sd
import soundfile as sf
import whisper
from TTS.api import TTS

from rex.assistant_errors import (
    SpeechToTextError,
    TextToSpeechError,
    WakeWordError,
)
from rex.config import AppConfig, load_config
from rex.llm_client import LanguageModel
from rex.logging_utils import get_logger
from rex.memory_utils import (
    append_history_entry,
    export_transcript,
    extract_voice_reference,
    load_all_profiles,
    load_users_map,
    resolve_user_key,
)
from rex.plugin_loader import load_plugins
from rex.wakeword_utils import detect_wakeword, load_wakeword_model

logger = get_logger(__name__)


@dataclass
class WakeWordListener:
    model: object
    threshold: float
    sample_rate: int
    block_size: int
    device: int | None
    loop: asyncio.AbstractEventLoop

    def __post_init__(self) -> None:
        self._event = asyncio.Event()
        self._stream: sd.InputStream | None = None

    def start(self) -> None:
        try:
            self._stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                callback=self._callback,
                device=self.device,
            )
            self._stream.start()
            logger.info("Wake-word listener started")
        except Exception as exc:
            raise WakeWordError(f"Failed to start input stream: {exc}")

    def stop(self) -> None:
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
                self._stream.close()
            logger.info("Wake-word listener stopped")
        self._event.set()

    def trigger(self) -> None:
        self.loop.call_soon_threadsafe(self._event.set)

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.warning("Audio status: %s", status)
        audio_data = np.squeeze(indata)
        try:
            if detect_wakeword(self.model, audio_data, threshold=self.threshold):
                self.loop.call_soon_threadsafe(self._event.set)
        except Exception as exc:
            logger.error("Wake-word detection failed: %s", exc)

    async def wait_for_wake(self) -> None:
        await self._event.wait()
        self._event.clear()


class AsyncRexAssistant:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.loop = asyncio.get_event_loop()
        self.language_model = LanguageModel(self.config)
        self._wake_model, self._wake_keyword = load_wakeword_model(keyword=self.config.wakeword)
        self._sample_rate = 16000
        self._listener = WakeWordListener(
            model=self._wake_model,
            threshold=self.config.wakeword_threshold,
            sample_rate=self._sample_rate,
            block_size=int(self._sample_rate * self.config.wakeword_window),
            device=self.config.audio_input_device,
            loop=self.loop,
        )
        self._tts: TTS | None = None
        self._whisper_model: whisper.Whisper | None = None

        self.users_map = load_users_map()
        self.profiles = load_all_profiles()
        resolved = resolve_user_key(
            self.config.default_user, self.users_map, profiles=self.profiles
        )
        self.active_user = resolved or (self.config.default_user or "james")

        self.user_voice_refs = {
            user: extract_voice_reference(profile, user_key=user)
            for user, profile in self.profiles.items()
        }

        self.plugins = load_plugins()
        logger.info("Loaded plugins: %s", ", ".join(self.plugins.keys()) or "none")

        self._running = True

    def _get_tts(self) -> TTS:
        if self._tts is None:
            self._tts = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
                gpu=False,
            )
        return self._tts

    def _get_whisper(self) -> whisper.Whisper:
        if self._whisper_model is None:
            try:
                self._whisper_model = whisper.load_model(self.config.whisper_model)
            except Exception as exc:
                raise SpeechToTextError(f"Failed to load Whisper model: {exc}")
        return self._whisper_model

    async def run(self) -> None:
        self._listener.start()
        try:
            while self._running:
                await self._listener.wait_for_wake()
                if not self._running:
                    break
                logger.info("Wake word '%s' detected", self._wake_keyword)
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
        if audio.size:
            min_val = float(np.min(audio))
            max_val = float(np.max(audio))
        else:
            min_val = max_val = 0.0
        logger.info("Audio shape: %s, dtype: %s, min: %.4f, max: %.4f", audio.shape, audio.dtype, min_val, max_val)

        try:
            transcript = await self.transcribe(audio, self._sample_rate)
        except SpeechToTextError as exc:
            logger.error("Transcription failed: %s", exc)
            return

        if not transcript:
            logger.info("No speech detected after wake word")
            return

        if self._maybe_switch_user(transcript):
            logger.info("Switched active user to %s", self.active_user)
            return

        logger.info("User (%s): %s", self.active_user, transcript)
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
            logger.error("TTS failed: %s", exc)

    def _record_audio(self) -> np.ndarray:
        sample_rate = self._sample_rate
        duration = self.config.command_duration
        frames = int(sample_rate * duration)
        logger.info("Recording %.1f seconds of audio", duration)
        audio = sd.rec(
            frames,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=self.config.audio_input_device,
        )
        sd.wait()
        audio = np.squeeze(audio)

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        logger.debug("[MIC] Recorded %s samples, %.2fs, peak volume %.4f", audio.size, duration, peak)

        debug_output = Path("debug_recording.wav")
        try:
            sf.write(debug_output, audio, sample_rate)
            logger.debug("[MIC] Saved debug audio to %s", debug_output)
        except Exception as exc:
            logger.warning("[MIC] Failed to save debug recording: %s", exc)

        return audio

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        self._get_whisper()

        def _transcribe() -> str:
            processed = audio.astype(np.float32) if audio.dtype != np.float32 else audio
            if processed.size:
                processed = np.clip(processed, -1.0, 1.0)
            result = self._whisper_model.transcribe(processed, language="en", fp16=False)
            logger.debug(f"[STT] Raw Whisper result: {result}")
            return str(result.get("text", "")).strip()

        try:
            return await asyncio.to_thread(_transcribe)
        except Exception as exc:
            logger.exception("[STT] Whisper failed:")
            raise SpeechToTextError(str(exc)) from exc

    def _speak_response(self, text: str) -> None:
        tts = self._get_tts()
        speaker_wav = self.user_voice_refs.get(self.active_user)
        try:
            tts.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language="en",
                file_path="assistant_response.wav",
            )
            wave_obj = sa.WaveObject.from_wave_file("assistant_response.wav")
            play_obj = wave_obj.play()
            play_obj.wait_done()
        except Exception as exc:
            raise TextToSpeechError(f"Failed to synthesise speech: {exc}")
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove("assistant_response.wav")

    def _play_wake_sound(self) -> None:
        path = self.config.wake_sound_path
        if not path:
            return
        try:
            wave_obj = sa.WaveObject.from_wave_file(path)
            play_obj = wave_obj.play()
            play_obj.wait_done()
        except Exception as exc:
            logger.warning("Failed to play wake sound %s: %s", path, exc)

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


def build_voice_loop(assistant: AsyncRexAssistant | None = None) -> AsyncRexAssistant:
    """Helper function for rex_loop.py to build and run the voice assistant loop."""
    return assistant or AsyncRexAssistant()
