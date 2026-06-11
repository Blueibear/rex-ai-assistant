"""Audio device validation, format conversion, and STT gain helpers — extracted verbatim from ``rex/voice_loop.py`` (US-REM-028)."""

from __future__ import annotations

import io
import wave
from contextvars import ContextVar
from typing import Any, cast

from rex.assistant_errors import (
    AudioDeviceError,
)
from rex.voice._types import (
    AudioArray,
)
from rex.voice.optional_imports import (
    _require_numpy,
    _require_sounddevice,
)


def _vl():
    """Return the ``rex.voice_loop`` facade module at call time.

    ``rex.voice_loop`` remains the single patch point for settings, lazy
    importers, audio helpers, and pipeline classes (tests monkeypatch
    ``rex.voice_loop.<name>``). Resolving through the facade at call time
    preserves that behavior without an import cycle at module load time.
    """
    import importlib

    return importlib.import_module("rex.voice_loop")


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


_STT_AUTO_GAIN_TARGET_PEAK = 0.45
_STT_AUTO_GAIN_MAX_GAIN = 12.0
_STT_AUTO_GAIN_MIN_RMS = 0.0005


def _audio_level(samples: AudioArray) -> tuple[float, float]:
    numpy = _require_numpy()
    if samples.size == 0:
        return 0.0, 0.0
    abs_samples = numpy.abs(samples)
    return (
        float(numpy.sqrt(numpy.mean(samples * samples))),
        float(numpy.max(abs_samples)),
    )


def _apply_stt_auto_gain(samples: AudioArray) -> AudioArray:
    numpy = _require_numpy()
    if not bool(getattr(_vl().settings, "stt_auto_gain", True)):
        return samples

    target_peak = float(getattr(_vl().settings, "stt_target_peak", _STT_AUTO_GAIN_TARGET_PEAK))
    max_gain = float(getattr(_vl().settings, "stt_max_gain", _STT_AUTO_GAIN_MAX_GAIN))
    min_rms = float(getattr(_vl().settings, "stt_min_rms_for_gain", _STT_AUTO_GAIN_MIN_RMS))
    rms, peak = _audio_level(samples)
    if rms < min_rms or peak <= 0.0 or peak >= target_peak:
        return samples

    gain = min(max_gain, target_peak / peak)
    boosted = numpy.clip(samples * gain, -1.0, 1.0)
    _vl().logger.info(
        "[STT] Applied input auto-gain",
        extra=_voice_log_extra(
            event="stt_audio_auto_gain",
            audio_rms_before=round(rms, 6),
            audio_peak_before=round(peak, 6),
            applied_gain=round(gain, 3),
            target_peak=round(target_peak, 3),
            max_gain=round(max_gain, 3),
        ),
    )
    return cast(AudioArray, boosted)


def _prepare_audio_for_stt(
    audio: AudioArray | bytes | bytearray | memoryview,
) -> AudioArray | bytes:
    """Return STT input with non-finite values removed and amplitude clamped.

    Whisper on CUDA can fail with NaN logits if the captured microphone buffer
    already contains NaN/inf values. Sanitize the audio before inference while
    preserving the preferred GPU execution path.
    """
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return bytes(audio)

    numpy = _require_numpy()
    prepared: AudioArray = numpy.asarray(audio, dtype=numpy.float32).reshape(-1)
    if prepared.size == 0:
        return prepared
    prepared = numpy.nan_to_num(prepared, nan=0.0, posinf=1.0, neginf=-1.0)
    prepared = numpy.clip(prepared, -1.0, 1.0)
    return _apply_stt_auto_gain(prepared)


def _audio_quality_summary(
    audio: AudioArray | bytes | bytearray | memoryview,
    sample_rate: int,
) -> dict[str, object]:
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return {
            "audio_input_kind": "bytes",
            "audio_bytes": len(bytes(audio)),
            "sample_rate": sample_rate,
        }

    numpy = _require_numpy()
    samples = numpy.asarray(audio, dtype=numpy.float32).reshape(-1)
    if samples.size == 0:
        return {
            "audio_input_kind": "array",
            "audio_samples": 0,
            "audio_duration_s": 0.0,
            "audio_rms": 0.0,
            "audio_peak": 0.0,
            "audio_clipped_samples": 0,
            "sample_rate": sample_rate,
        }

    abs_samples = numpy.abs(samples)
    return {
        "audio_input_kind": "array",
        "audio_samples": int(samples.size),
        "audio_duration_s": round(samples.size / sample_rate, 3) if sample_rate > 0 else None,
        "audio_rms": round(float(numpy.sqrt(numpy.mean(samples * samples))), 6),
        "audio_peak": round(float(numpy.max(abs_samples)), 6),
        "audio_clipped_samples": int(numpy.count_nonzero(abs_samples >= 0.999)),
        "sample_rate": sample_rate,
    }


_VOICE_INTERACTION_ID: ContextVar[int | None] = ContextVar(
    "rex_voice_interaction_id",
    default=None,
)


def _voice_log_extra(**extra: object) -> dict[str, object]:
    interaction_id = _VOICE_INTERACTION_ID.get()
    if interaction_id is not None and "interaction_id" not in extra:
        extra["interaction_id"] = interaction_id
    return extra
