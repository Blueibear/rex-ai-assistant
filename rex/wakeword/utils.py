"""Picovoice Porcupine wake-word integration and helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:  # pragma: no cover - optional runtime dependency
    import pvporcupine
except ImportError:  # pragma: no cover - Porcupine not installed
    pvporcupine = None  # type: ignore

from config import load_config
from rex.assistant_errors import WakeWordError
from rex.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class PorcupineDetector:
    """Stateful wrapper that accumulates PCM frames for Porcupine."""

    porcupine: "pvporcupine.Porcupine"
    keyword: str
    sensitivity: float

    def __post_init__(self) -> None:
        self._buffer = np.zeros(0, dtype=np.int16)

    @property
    def frame_length(self) -> int:
        return self.porcupine.frame_length

    @property
    def sample_rate(self) -> int:
        return self.porcupine.sample_rate

    def push(self, audio_frame: np.ndarray) -> bool:
        """Ingest audio samples and evaluate for a wake-word hit."""

        if audio_frame.ndim != 1:
            audio_frame = np.reshape(audio_frame, (-1,))

        if audio_frame.dtype != np.int16:
            scaled = np.clip(audio_frame, -1.0, 1.0)
            audio_frame = (scaled * np.iinfo(np.int16).max).astype(np.int16)

        if self._buffer.size:
            audio_frame = np.concatenate((self._buffer, audio_frame))
            self._buffer = np.zeros(0, dtype=np.int16)

        frame_length = self.porcupine.frame_length
        total_samples = audio_frame.shape[0]
        index = 0

        while index + frame_length <= total_samples:
            frame = audio_frame[index : index + frame_length]
            result = self.porcupine.process(frame)
            if result >= 0:
                LOGGER.debug("Porcupine detected keyword '%s' (index=%s)", self.keyword, result)
                return True
            index += frame_length

        if index < total_samples:
            self._buffer = audio_frame[index:]
        return False

    def close(self) -> None:
        self.porcupine.delete()


def _resolve_keyword(keyword: Optional[str]) -> str:
    cfg = load_config()
    if keyword:
        return keyword
    return cfg.wakeword


def load_wakeword_model(
    *, keyword: Optional[str] = None, sensitivity: Optional[float] = None, keyword_path: Optional[str] = None
) -> tuple[PorcupineDetector, str]:
    """Instantiate a Porcupine detector using config defaults."""

    if pvporcupine is None:  # pragma: no cover - ensures helpful runtime message
        raise WakeWordError("pvporcupine is not installed. Please install it via requirements.txt")

    cfg = load_config()
    resolved_keyword = _resolve_keyword(keyword)
    sensitivity_value = sensitivity if sensitivity is not None else cfg.wakeword_sensitivity
    sensitivity_value = max(0.1, min(0.99, float(sensitivity_value)))

    access_key = os.getenv("PICOVOICE_ACCESS_KEY")

    kwargs = {
        "access_key": access_key,
        "keywords": [resolved_keyword] if keyword_path is None else None,
        "keyword_paths": [keyword_path] if keyword_path is not None else None,
        "sensitivities": [sensitivity_value],
    }
    try:
        porcupine = pvporcupine.create(**kwargs)
    except TypeError:
        # Older versions do not accept None keyword_paths/keywords simultaneously
        if keyword_path:
            porcupine = pvporcupine.create(access_key=access_key, keyword_paths=[keyword_path], sensitivities=[sensitivity_value])
        else:
            porcupine = pvporcupine.create(access_key=access_key, keywords=[resolved_keyword], sensitivities=[sensitivity_value])
    except Exception as exc:  # pragma: no cover - initialization failure depends on runtime
        raise WakeWordError(f"Failed to initialize Porcupine: {exc}") from exc

    detector = PorcupineDetector(porcupine=porcupine, keyword=resolved_keyword, sensitivity=sensitivity_value)
    LOGGER.info(
        "Initialized Porcupine wake-word detector keyword='%s' sensitivity=%.2f sample_rate=%d frame_length=%d",
        resolved_keyword,
        sensitivity_value,
        detector.sample_rate,
        detector.frame_length,
    )
    return detector, resolved_keyword


def detect_wakeword(model: PorcupineDetector, audio_frame: np.ndarray, *, threshold: float | None = None) -> bool:
    """Compatibility shim for historical signature. Threshold is unused."""

    _ = threshold  # maintained for backward compatibility
    return model.push(audio_frame)


__all__ = ["PorcupineDetector", "load_wakeword_model", "detect_wakeword"]
