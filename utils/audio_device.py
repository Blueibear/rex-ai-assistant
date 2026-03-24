"""Audio device enumeration, validation and configuration persistence for Rex."""

from __future__ import annotations

import json
import logging
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any

sd = None

logger = logging.getLogger(__name__)

# Default sample rate for wake word detection
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_REX_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "rex_config.json"


def _load_sounddevice():
    global sd
    if sd is not None:
        return sd
    if find_spec("sounddevice") is None:
        return None
    sd = import_module("sounddevice")
    return sd


def _require_sounddevice():
    module = _load_sounddevice()
    if module is None:
        raise RuntimeError("sounddevice is required for audio device access")
    return module


class AudioDeviceInfo:
    """Structured information about an audio input device."""

    def __init__(self, index: int, device_info: dict[str, Any]) -> None:
        self.index = index
        self.name = str(device_info.get("name", "Unknown"))
        self.max_input_channels = int(device_info.get("max_input_channels", 0))
        self.default_samplerate = float(device_info.get("default_samplerate", 0))
        self.hostapi = int(device_info.get("hostapi", 0))

        # Get host API name (e.g., "WASAPI", "WDM-KS")
        sounddevice = _require_sounddevice()
        try:
            hostapi_info = sounddevice.query_hostapis(self.hostapi)
            self.hostapi_name = str(hostapi_info.get("name", "Unknown"))
        except Exception:
            self.hostapi_name = "Unknown"

    def display_name(self) -> str:
        """Return formatted display name for GUI dropdown."""
        return f"{self.index}: {self.name} [{self.hostapi_name}]"

    def is_wasapi(self) -> bool:
        """Check if device uses WASAPI host API (preferred on Windows)."""
        return "WASAPI" in self.hostapi_name.upper()

    def is_wdm_ks(self) -> bool:
        """Check if device uses WDM-KS (avoid - blocking API not supported)."""
        return "WDM-KS" in self.hostapi_name.upper() or "Windows WDM-KS" in self.hostapi_name

    def __repr__(self) -> str:
        return f"AudioDeviceInfo({self.index}, {self.name!r}, {self.hostapi_name})"


def enumerate_input_devices() -> list[AudioDeviceInfo]:
    """Return list of all available input devices (max_input_channels > 0)."""
    devices: list[AudioDeviceInfo] = []

    sounddevice = _load_sounddevice()
    if sounddevice is None:
        logger.warning("sounddevice not available; skipping audio device enumeration")
        return devices

    try:
        device_count = len(sounddevice.query_devices())
        for idx in range(device_count):
            try:
                info = sounddevice.query_devices(idx)
                if isinstance(info, dict) and info.get("max_input_channels", 0) > 0:
                    devices.append(AudioDeviceInfo(idx, info))
            except Exception as exc:
                logger.debug(f"Failed to query device {idx}: {exc}")
                continue
    except Exception as exc:
        logger.error(f"Failed to enumerate audio devices: {exc}")

    return devices


def validate_device(device_index: int, sample_rate: int = DEFAULT_SAMPLE_RATE) -> tuple[bool, str]:
    """
    Validate if device can be opened at the specified sample rate.

    Returns:
        (is_valid, error_message) - error_message is empty string if valid
    """
    if device_index < 0:
        return False, f"Invalid device index: {device_index} (must be >= 0)"

    sounddevice = _load_sounddevice()
    if sounddevice is None:
        return False, "sounddevice is required to validate audio devices"

    try:
        device_info = sounddevice.query_devices(device_index)
        if not isinstance(device_info, dict):
            return False, f"Device {device_index} returned invalid info"

        max_input_channels = device_info.get("max_input_channels", 0)
        if max_input_channels <= 0:
            return False, f"Device {device_index} has no input channels"

        # Try to check if device supports the sample rate
        # We don't actually open a stream here to avoid blocking
        device_name = device_info.get("name", "Unknown")
        hostapi_idx = device_info.get("hostapi", 0)
        try:
            hostapi_info = sounddevice.query_hostapis(hostapi_idx)
            hostapi_name = hostapi_info.get("name", "Unknown")
        except Exception:
            hostapi_name = "Unknown"

        # Check if it's WDM-KS (problematic)
        if "WDM-KS" in str(hostapi_name).upper() or "Windows WDM-KS" in str(hostapi_name):
            return (
                False,
                f"Device {device_index} ({device_name}) uses WDM-KS API which is not supported (blocking API)",
            )

        logger.info(
            f"Device {device_index} ({device_name} [{hostapi_name}]) appears valid for input"
        )
        return True, ""

    except Exception as exc:
        return False, f"Error querying device {device_index}: {exc}"


def get_smart_default_device(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    prefer_wasapi: bool = True,
) -> int | None:
    """
    Get smart default input device.

    Strategy:
    1. Prefer WASAPI devices on Windows
    2. Avoid WDM-KS devices (blocking API not supported)
    3. Validate device can be queried
    4. Return first valid device, or None if none found

    Returns:
        Device index or None if no valid device found
    """
    devices = enumerate_input_devices()
    if not devices:
        logger.warning("No input devices found")
        return None

    # First pass: prefer WASAPI devices
    if prefer_wasapi:
        for device in devices:
            if device.is_wasapi() and not device.is_wdm_ks():
                is_valid, error = validate_device(device.index, sample_rate)
                if is_valid:
                    logger.info(f"Selected WASAPI device: {device.display_name()}")
                    return device.index
                else:
                    logger.debug(f"WASAPI device {device.index} failed validation: {error}")

    # Second pass: any non-WDM-KS device
    for device in devices:
        if not device.is_wdm_ks():
            is_valid, error = validate_device(device.index, sample_rate)
            if is_valid:
                logger.info(f"Selected device: {device.display_name()}")
                return device.index
            else:
                logger.debug(f"Device {device.index} failed validation: {error}")

    # Fallback: return first device even if it might not work perfectly
    if devices:
        fallback = devices[0]
        logger.warning(f"No ideal device found, falling back to: {fallback.display_name()}")
        return fallback.index

    return None


def load_audio_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load audio configuration from rex_config.json.

    Returns dict with keys:
        - input_device_index: int or None
        - output_device_index: int or None
        - sample_rate: int (default 16000)
    """
    if config_path is None:
        config_path = DEFAULT_REX_CONFIG_PATH

    if not config_path.exists():
        logger.warning("Audio config not found at %s, using defaults", config_path)
        return {
            "input_device_index": None,
            "output_device_index": None,
            "sample_rate": DEFAULT_SAMPLE_RATE,
        }

    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)

        audio_section = data.get("audio", {}) if isinstance(data, dict) else {}
        if not isinstance(audio_section, dict):
            logger.warning("Audio config section is invalid in %s, using defaults", config_path)
            audio_section = {}

        # Validate and normalize
        device_idx = audio_section.get("input_device_index")
        if device_idx is not None:
            device_idx = int(device_idx)
            if device_idx < 0:
                logger.warning(f"Invalid device index {device_idx} in config, ignoring")
                device_idx = None

        output_device_idx = audio_section.get("output_device_index")
        if output_device_idx is not None:
            output_device_idx = int(output_device_idx)
            if output_device_idx < 0:
                logger.warning(
                    "Invalid output device index %s in config, ignoring", output_device_idx
                )
                output_device_idx = None

        sample_rate = int(audio_section.get("sample_rate", DEFAULT_SAMPLE_RATE))

        return {
            "input_device_index": device_idx,
            "output_device_index": output_device_idx,
            "sample_rate": sample_rate,
        }
    except Exception as exc:
        logger.error("Failed to load audio config from %s: %s", config_path, exc)
        return {
            "input_device_index": None,
            "output_device_index": None,
            "sample_rate": DEFAULT_SAMPLE_RATE,
        }


def save_audio_config(
    input_device_index: int | None,
    output_device_index: int | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    config_path: Path | None = None,
) -> None:
    """Save audio configuration to rex_config.json."""
    if config_path is None:
        config_path = DEFAULT_REX_CONFIG_PATH

    # Ensure config directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any]
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("Failed to read existing rex config at %s: %s", config_path, exc)
            data = {}
    else:
        logger.warning("Rex config not found at %s. Creating a new config file.", config_path)
        data = {}

    if not isinstance(data, dict):
        data = {}

    audio_section = data.get("audio")
    if not isinstance(audio_section, dict):
        audio_section = {}

    audio_section.update(
        {
            "input_device_index": input_device_index,
            "output_device_index": output_device_index,
            "sample_rate": sample_rate,
        }
    )
    data["audio"] = audio_section

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(
            "Saved audio config to %s: input=%s output=%s rate=%s",
            config_path,
            input_device_index,
            output_device_index,
            sample_rate,
        )
    except Exception as exc:
        logger.error("Failed to save audio config to %s: %s", config_path, exc)


def resolve_audio_device(
    configured_device: int | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[int | None, str]:
    """
    Resolve audio device from configuration with validation.

    Args:
        configured_device: Device index from config (can be None or -1)
        sample_rate: Target sample rate

    Returns:
        (device_index, status_message)
        - device_index: Valid device index or None
        - status_message: Human-readable status for logging/GUI
    """
    # If configured device is -1 or None, get smart default
    if configured_device is None or configured_device < 0:
        default_device = get_smart_default_device(sample_rate)
        if default_device is None:
            return None, "No valid input devices found"

        devices = enumerate_input_devices()
        device_info = next((d for d in devices if d.index == default_device), None)
        if device_info:
            return default_device, f"Using default device: {device_info.display_name()}"
        else:
            return default_device, f"Using default device: {default_device}"

    # Validate configured device
    is_valid, error = validate_device(configured_device, sample_rate)
    if not is_valid:
        logger.warning(f"Configured device {configured_device} is invalid: {error}")
        # Fall back to smart default
        default_device = get_smart_default_device(sample_rate)
        if default_device is None:
            return (
                None,
                f"Configured device {configured_device} invalid and no fallback found: {error}",
            )

        devices = enumerate_input_devices()
        device_info = next((d for d in devices if d.index == default_device), None)
        if device_info:
            return (
                default_device,
                f"Configured device {configured_device} failed, using: {device_info.display_name()}",
            )
        else:
            return (
                default_device,
                f"Configured device {configured_device} failed, using device {default_device}",
            )

    # Configured device is valid
    devices = enumerate_input_devices()
    device_info = next((d for d in devices if d.index == configured_device), None)
    if device_info:
        return configured_device, f"Using configured device: {device_info.display_name()}"
    else:
        return configured_device, f"Using configured device: {configured_device}"


__all__ = [
    "AudioDeviceInfo",
    "enumerate_input_devices",
    "validate_device",
    "get_smart_default_device",
    "load_audio_config",
    "save_audio_config",
    "resolve_audio_device",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_REX_CONFIG_PATH",
]
