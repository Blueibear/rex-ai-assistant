#!/usr/bin/env python3
"""Async event loop orchestrating Rex's wake-word, STT, LLM and TTS pipeline."""

from __future__ import annotations

# CRITICAL: Apply transformers compatibility patch BEFORE importing TTS
# Step 1: Import and apply the compatibility shim
from rex.compat import ensure_transformers_compatibility
ensure_transformers_compatibility()  # Explicitly call to ensure it runs

# Step 2: Force transformers to load NOW with the patch applied
import transformers  # noqa: F401

# Step 3: NOW it's safe to import TTS (which will use patched transformers)
import asyncio
import contextlib
import os
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from TTS.api import TTS

try:
    import simpleaudio as sa  # type: ignore
except ImportError:
    sa = None

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
from utils.audio_device import (
    enumerate_input_devices,
    load_audio_config,
    resolve_audio_device,
)

logger = get_logger(__name__)


@dataclass
class WakeWordListener:
    model: object
    threshold: float
    sample_rate: int
    block_size: int
    device: int | None
    loop: asyncio.AbstractEventLoop | None = None

    def __post_init__(self) -> None:
        self._event = asyncio.Event()
        self._stream: sd.InputStream | None = None
        self._callback_count = 0  # Track number of audio callbacks received

    def start(self) -> None:
        """Start audio input stream with robust Windows-friendly fallback strategy."""
        # Get device info for logging and error messages
        device_name = "default"
        device_hostapi = "default"
        if self.device is not None:
            try:
                devices = enumerate_input_devices()
                device_obj = next((d for d in devices if d.index == self.device), None)
                if device_obj:
                    device_name = device_obj.name
                    device_hostapi = device_obj.hostapi_name
                else:
                    device_name = f"device_{self.device}"
            except Exception:
                device_name = f"device_{self.device}"

        # Get device default samplerate if available
        device_default_sr = None
        if self.device is not None:
            try:
                info = sd.query_devices(self.device)
                if isinstance(info, dict):
                    device_default_sr = int(info.get("default_samplerate", 0))
            except Exception:
                pass

        # Define retry strategies: (samplerate, blocksize, latency)
        # Try different combinations for Windows WASAPI compatibility
        retry_strategies = []

        # Start with requested configuration
        retry_strategies.append((self.sample_rate, self.block_size, None))

        # Try with device's default samplerate
        if device_default_sr and device_default_sr != self.sample_rate:
            retry_strategies.append((device_default_sr, self.block_size, None))

        # Try common samplerates
        for sr in [16000, 44100, 48000]:
            if sr != self.sample_rate and (not device_default_sr or sr != device_default_sr):
                retry_strategies.append((sr, self.block_size, None))

        # Try with different blocksizes
        retry_strategies.append((self.sample_rate, None, None))
        retry_strategies.append((self.sample_rate, 1024, None))
        retry_strategies.append((self.sample_rate, 2048, None))

        # Try with high latency
        retry_strategies.append((self.sample_rate, self.block_size, "high"))

        # Attempt to start stream with retry strategies
        last_exception = None
        attempt_details = []

        for attempt_num, (samplerate, blocksize, latency) in enumerate(retry_strategies, 1):
            try:
                logger.info(
                    f"Attempt {attempt_num}: Using input device index: {self.device} "
                    f"name: {device_name} hostapi: {device_hostapi} "
                    f"samplerate: {samplerate} channels: 1 dtype: float32 "
                    f"blocksize: {blocksize} latency: {latency or 'default'}"
                )

                stream_kwargs = {
                    "channels": 1,
                    "samplerate": samplerate,
                    "callback": self._callback,
                    "device": self.device,
                    "dtype": "float32",
                }

                if blocksize is not None:
                    stream_kwargs["blocksize"] = blocksize

                if latency is not None:
                    stream_kwargs["latency"] = latency

                self._stream = sd.InputStream(**stream_kwargs)
                self._stream.start()

                # Success!
                logger.info(f"Wake-word listener started successfully on device {self.device} ({device_name}) at {samplerate} Hz")

                # Update sample_rate if we had to use a different one
                if samplerate != self.sample_rate:
                    logger.warning(f"Using samplerate {samplerate} instead of requested {self.sample_rate}")
                    self.sample_rate = samplerate

                return

            except Exception as exc:
                last_exception = exc
                attempt_details.append(
                    f"  Attempt {attempt_num} [sr={samplerate}, bs={blocksize}, lat={latency or 'default'}]: {type(exc).__name__}: {exc}"
                )
                logger.debug(f"Stream start attempt {attempt_num} failed: {exc}")

                # Close stream if it was partially created
                if self._stream is not None:
                    try:
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None

        # All attempts failed - build comprehensive error message
        error_msg = f"Failed to start input stream on device {self.device} ({device_name}) [{device_hostapi}]\n"
        error_msg += f"Tried {len(retry_strategies)} different configurations:\n"
        error_msg += "\n".join(attempt_details)
        error_msg += f"\n\nFinal error: {last_exception}"

        # Add helpful suggestions for common Windows DirectSound errors
        error_str = str(last_exception).lower()
        if "directsound" in error_str and "-2005401480" in error_str:
            error_msg += "\n\n💡 DirectSound Exclusive Access Error - Try these fixes:"
            error_msg += "\n   1. Close any apps using the microphone (Skype, Teams, Discord, Zoom, OBS)"
            error_msg += "\n   2. Disable exclusive mode: Sound settings → Recording → Device Properties → Advanced → Uncheck 'Allow exclusive control'"
            error_msg += "\n   3. Look for a WASAPI version of this device in the device dropdown (more reliable than DirectSound)"
            error_msg += "\n   4. Try a different audio device"
        elif "directsound" in error_str or "wasapi" in error_str:
            error_msg += "\n\n💡 Windows Audio Error - Suggestions:"
            error_msg += "\n   1. Close other audio applications"
            error_msg += "\n   2. Try selecting a different device"
            error_msg += "\n   3. Check Windows Sound settings → Recording tab"

        raise WakeWordError(error_msg)

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
        self._callback_count += 1

        # Log every 100 callbacks (~5-10 seconds) to show we're receiving audio
        if self._callback_count % 100 == 1:
            audio_level = np.max(np.abs(indata)) if indata.size > 0 else 0.0
            logger.info(f"Audio callback #{self._callback_count}: receiving audio (level: {audio_level:.3f})")

        if status:
            logger.warning("Audio status: %s", status)

        audio_data = np.squeeze(indata)

        # TEMPORARY DEBUG: Log before calling detect_wakeword
        if self._callback_count % 100 == 1:
            logger.info(f">>> ABOUT TO CALL detect_wakeword (callback #{self._callback_count}, threshold={self.threshold})")

        try:
            result = detect_wakeword(self.model, audio_data, threshold=self.threshold)

            # TEMPORARY DEBUG: Log after calling detect_wakeword
            if self._callback_count % 100 == 1:
                logger.info(f"<<< detect_wakeword RETURNED: {result}")

            if result:
                logger.info(f"!!! WAKE WORD DETECTED IN CALLBACK - calling loop.call_soon_threadsafe to set event")
                if self.loop is None:
                    logger.error("!!! ERROR: loop is None - cannot schedule event.set()!")
                else:
                    self.loop.call_soon_threadsafe(self._event.set)
                    logger.info(f"!!! Event.set() scheduled on event loop: {self.loop}")
        except Exception as exc:
            logger.error("Wake-word detection failed: %s", exc)
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    async def wait_for_wake(self) -> None:
        logger.info(">>> wait_for_wake() called - waiting for event...")
        await self._event.wait()
        logger.info("<<< wait_for_wake() - event received! Clearing event...")
        self._event.clear()
        logger.info("<<< wait_for_wake() - event cleared, returning")


class AsyncRexAssistant:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.language_model = LanguageModel(self.config)
        self._wake_model, self._wake_keyword = load_wakeword_model(keyword=self.config.wakeword)
        self._sample_rate = 16000

        # Load audio config and resolve device with validation
        audio_config = load_audio_config()
        configured_device = audio_config.get("input_device_index") or self.config.audio_input_device
        resolved_device, status_msg = resolve_audio_device(configured_device, self._sample_rate)

        if resolved_device is None:
            logger.warning(f"Audio device resolution failed: {status_msg}")
            logger.warning("Will attempt to use default device (None)")
        else:
            logger.info(status_msg)

        # Create listener with a placeholder loop (will be updated in run() with the actual running loop)
        self._listener = WakeWordListener(
            model=self._wake_model,
            threshold=self.config.wakeword_threshold,
            sample_rate=self._sample_rate,
            block_size=int(self._sample_rate * self.config.wakeword_window),
            device=resolved_device,
            loop=None,  # Will be set to the running loop in run()
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
            import torch
            gpu_available = torch.cuda.is_available()
            logger.info(f"Loading TTS model (XTTS v2) on {'GPU' if gpu_available else 'CPU (WARNING: GPU not available!)'}...")
            self._tts = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
                gpu=gpu_available,  # Enable GPU if available, fallback to CPU
            )
            logger.info("TTS model loaded successfully")
        return self._tts

    def _get_whisper(self) -> whisper.Whisper:
        if self._whisper_model is None:
            try:
                self._whisper_model = whisper.load_model(self.config.whisper_model)
            except Exception as exc:
                raise SpeechToTextError(f"Failed to load Whisper model: {exc}")
        return self._whisper_model

    async def run(self) -> None:
        # Log wake word details right before starting (this is in background thread so GUI will see it)
        logger.info(f"Starting wake word listener for keyword: '{self._wake_keyword}' (threshold: {self.config.wakeword_threshold})")
        logger.info(f"Wake word model type: {type(self._wake_model).__name__}")

        # CRITICAL: Get the running event loop and update listener to use it
        # This ensures the audio callback schedules events on the correct loop
        running_loop = asyncio.get_running_loop()
        self._listener.loop = running_loop
        logger.info(f"Updated listener to use running event loop: {running_loop}")

        # Pre-load models in background thread for faster first response
        logger.info("Pre-loading Whisper and TTS models in background...")
        await asyncio.to_thread(self._get_whisper)  # Pre-load Whisper
        await asyncio.to_thread(self._get_tts)      # Pre-load TTS on GPU
        logger.info("Models pre-loaded successfully")

        self._listener.start()
        try:
            logger.info(">>> Entering main run loop - waiting for wake word...")
            while self._running:
                logger.info(">>> Loop iteration - calling wait_for_wake()...")
                await self._listener.wait_for_wake()
                logger.info(">>> wait_for_wake() returned!")
                if not self._running:
                    logger.info(">>> _running is False, breaking loop")
                    break
                logger.info("Wake word '%s' detected", self._wake_keyword)
                logger.info(">>> Calling _handle_interaction()...")
                await self._handle_interaction()
                logger.info(">>> _handle_interaction() completed")
        finally:
            self._listener.stop()

    async def _handle_interaction(self) -> None:
        import time
        start_time = time.time()
        logger.info(">>> _handle_interaction() started - playing wake sound and processing conversation...")
        try:
            # Play wake sound first, then record (don't overlap to avoid audio device conflicts)
            await asyncio.to_thread(self._play_wake_sound)
            await self._process_conversation()
            total_time = time.time() - start_time
            logger.info(f">>> _handle_interaction() completed successfully in {total_time:.2f} seconds")
        except Exception as exc:
            logger.error(f"!!! _handle_interaction() FAILED: {exc}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    async def _process_conversation(self) -> None:
        logger.info(">>> _process_conversation() started - recording audio...")
        audio = await asyncio.to_thread(self._record_audio)
        logger.info(f">>> Audio recording complete - {audio.size} samples")
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

        # Use the same device as the listener
        device = self._listener.device

        audio = sd.rec(
            frames,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device,
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
        import time
        tts = self._get_tts()
        speaker_wav = self.user_voice_refs.get(self.active_user)
        try:
            # XTTS v2 requires either speaker_wav or speaker parameter
            # If no voice reference, use built-in speaker
            logger.info(f"[TTS] Generating speech for {len(text)} characters...")
            tts_start = time.time()

            if speaker_wav:
                tts.tts_to_file(
                    text=text,
                    speaker_wav=speaker_wav,
                    language="en",
                    file_path="assistant_response.wav",
                )
            else:
                # Use built-in XTTS speaker when no voice reference is available
                tts.tts_to_file(
                    text=text,
                    speaker="Claribel Dervla",  # Default XTTS v2 speaker
                    language="en",
                    file_path="assistant_response.wav",
                )

            tts_duration = time.time() - tts_start
            logger.info(f"[TTS] Speech generated in {tts_duration:.2f} seconds")

            # Play the generated audio
            logger.info("[TTS] Playing audio...")
            playback_start = time.time()

            if platform.system() == "Windows":
                # Use winsound on Windows (more reliable than simpleaudio)
                import winsound
                winsound.PlaySound("assistant_response.wav", winsound.SND_FILENAME)
            elif sa is not None:
                # Use simpleaudio on other platforms
                wave_obj = sa.WaveObject.from_wave_file("assistant_response.wav")
                play_obj = wave_obj.play()
                play_obj.wait_done()
            else:
                logger.warning("No audio playback library available - audio saved but not played")

            playback_duration = time.time() - playback_start
            logger.info(f"[TTS] Audio playback completed in {playback_duration:.2f} seconds")
        except Exception as exc:
            raise TextToSpeechError(f"Failed to synthesise speech: {exc}")
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove("assistant_response.wav")

    def _play_wake_sound(self) -> None:
        path = self.config.wake_sound_path
        if not path:
            logger.debug("No wake sound path configured - skipping acknowledgment tone")
            return

        # Check if file exists
        if not Path(path).exists():
            logger.warning(f"Wake sound file not found: {path}")
            return

        # Use platform-specific playback for reliability
        if platform.system() == "Windows":
            self._play_wake_sound_windows(path)
        else:
            self._play_wake_sound_sounddevice(path)

    def _play_wake_sound_windows(self, path: str | Path) -> None:
        """Play wake sound on Windows using winsound (more reliable than sounddevice)."""
        try:
            import winsound
            logger.debug(f"Playing wake sound (Windows): {path}")

            # First, stop any currently playing sound
            winsound.PlaySound(None, winsound.SND_PURGE)

            # Play the wake sound synchronously (blocks until complete)
            # SND_FILENAME: path is a file
            # SND_NODEFAULT: no default beep if file missing
            # SND_NOSTOP: don't interrupt if another sound is playing (we already purged)
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_NODEFAULT)

            # Ensure playback is fully stopped
            winsound.PlaySound(None, winsound.SND_PURGE)

            logger.info("Wake acknowledgment tone played (Windows)")
        except Exception as exc:
            logger.warning("Failed to play wake sound with winsound: %s, falling back to sounddevice", exc)
            self._play_wake_sound_sounddevice(path)

    def _play_wake_sound_sounddevice(self, path: str | Path) -> None:
        """Play wake sound using sounddevice (cross-platform fallback)."""
        try:
            # Read audio file
            data, samplerate = sf.read(path)
            logger.debug(f"Playing wake sound: {path} ({len(data)} samples at {samplerate} Hz)")

            # Forcefully abort any existing streams before playing
            try:
                sd.stop()
            except Exception:
                pass

            # Use blocking playback to ensure proper cleanup
            device = self.config.audio_output_device
            sd.play(data, samplerate, device=device, blocking=True)

            # Ensure stream is fully stopped
            sd.stop()

            logger.info("Wake acknowledgment tone played")
        except Exception as exc:
            logger.warning("Failed to play wake sound %s: %s", path, exc)
            # Forcefully stop any stuck playback
            try:
                sd.stop()
            except Exception:
                pass


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

