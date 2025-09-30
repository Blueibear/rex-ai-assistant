"""Compatibility helpers for wake-word detection with graceful fallbacks."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from assistant_errors import WakeWordError

DEFAULT_THRESHOLD = 0.5


def _extract_score(result: Any) -> Optional[float]:
    if isinstance(result, (int, float)):
        return float(result)
    if isinstance(result, dict):
        for value in result.values():
            if isinstance(value, (int, float)):
                return float(value)
        return None
    if isinstance(result, Iterable) and not isinstance(result, (str, bytes)):
        for value in result:
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _legacy_predict_score(model: Any, audio_frame: Any) -> float:
    if hasattr(model, "predict"):
        result = model.predict(audio_frame)
    elif callable(model):
        result = model(audio_frame)
    else:
        raise WakeWordError(
            "Wake-word model must implement either push(), predict(), or be callable to score audio."
        )
    score = _extract_score(result)
    if score is None:
        raise WakeWordError("Wake-word detector did not return a numeric score.")
    return score


try:  # pragma: no cover - prefer the full-featured implementation when available
    from rex.wakeword.utils import PorcupineDetector as _PorcupineDetector
    from rex.wakeword.utils import detect_wakeword as _porcupine_detect
    from rex.wakeword.utils import load_wakeword_model
except ImportError as exc:  # pragma: no cover - triggered when optional deps missing
    _IMPORT_ERROR = exc

    class PorcupineDetector:  # type: ignore[override]
        """Placeholder used when Porcupine dependencies are unavailable."""

        sample_rate: int = 16000
        frame_length: int = 512

        def push(self, _audio_frame: Any) -> bool:  # pragma: no cover - simple stub
            raise WakeWordError(
                "Porcupine dependencies are missing. Install pvporcupine and numpy to enable wake-word detection."
            ) from _IMPORT_ERROR

        def close(self) -> None:  # pragma: no cover - simple stub
            return None

    def load_wakeword_model(
        *, keyword: Optional[str] = None, sensitivity: Optional[float] = None, keyword_path: Optional[str] = None
    ) -> tuple[PorcupineDetector, str]:  # pragma: no cover - simple stub
        raise WakeWordError(
            "Porcupine dependencies are missing. Install pvporcupine and numpy to enable wake-word detection."
        ) from _IMPORT_ERROR

    def detect_wakeword(model: Any, audio_frame: Any, *, threshold: float | None = None) -> bool:
        score = _legacy_predict_score(model, audio_frame)
        cutoff = threshold if threshold is not None else DEFAULT_THRESHOLD
        return score >= cutoff

else:

    class PorcupineDetector(_PorcupineDetector):  # pragma: no cover - behaviour provided by upstream
        pass

    def detect_wakeword(model: Any, audio_frame: Any, *, threshold: float | None = None) -> bool:
        if hasattr(model, "push"):
            return _porcupine_detect(model, audio_frame, threshold=threshold)
        score = _legacy_predict_score(model, audio_frame)
        cutoff = threshold if threshold is not None else DEFAULT_THRESHOLD
        return score >= cutoff


__all__ = ["PorcupineDetector", "load_wakeword_model", "detect_wakeword"]
