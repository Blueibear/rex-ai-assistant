"""Helpers for loading wake-word models and evaluating predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple, TYPE_CHECKING

try:  # pragma: no cover - optional heavy dependency
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover - used only when available
    _np = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    import numpy as np  # type: ignore
    NDArray = np.ndarray
else:
    NDArray = Any

try:  # pragma: no cover - optional model dependency
    import openwakeword
    from openwakeword.model import Model as WakeWordModel
except ModuleNotFoundError:  # pragma: no cover - degrade gracefully
    openwakeword = None  # type: ignore

    class WakeWordModel:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - defensive
            raise RuntimeError("openwakeword is not installed")

    _AVAILABLE_KEYWORDS: set[str] = set()
else:
    _AVAILABLE_KEYWORDS = {name.replace("_", " ") for name in openwakeword.MODELS.keys()}

from ..config import settings
from ..logging_utils import configure_logger

LOGGER = configure_logger(__name__)
_DEFAULT_BACKEND = "onnx"


def _resolve_model_path(model_path: str | None = None) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if model_path:
        return Path(model_path)
    return repo_root / "rex.onnx"


def load_wakeword_model(
    *, keyword: str | None = None, model_path: str | None = None
) -> Tuple[WakeWordModel, str]:
    """Return a configured openWakeWord model and the active wake phrase."""

    if openwakeword is None:
        raise RuntimeError("openwakeword is required to load wake word models")

    resolved_path = _resolve_model_path(model_path)
    requested_keyword = (keyword or settings.wakeword).replace("_", " ")

    if resolved_path.is_file() and resolved_path.stat().st_size > 0:
        models_to_load = [str(resolved_path)]
        active_label = resolved_path.stem
    else:
        if _AVAILABLE_KEYWORDS and requested_keyword not in _AVAILABLE_KEYWORDS:
            LOGGER.warning(
                "Keyword '%s' not available; falling back to default '%s'",
                requested_keyword,
                settings.wakeword,
            )
            requested_keyword = settings.wakeword.replace("_", " ")
        models_to_load = [requested_keyword]
        active_label = requested_keyword

    wake_model = WakeWordModel(
        wakeword_models=models_to_load,
        inference_framework=_DEFAULT_BACKEND,
        enable_speex_noise_suppression=True,
    )

    return wake_model, active_label


def detect_wakeword(
    model: WakeWordModel,
    audio_frame: NDArray,
    *,
    threshold: float | None = None,
) -> bool:
    """Return ``True`` when ``audio_frame`` contains the active wake word."""

    if _np is None:
        raise RuntimeError("numpy is required for wake word detection")

    if getattr(audio_frame, "ndim", 1) != 1:
        audio_frame = _np.reshape(audio_frame, (-1,))

    if getattr(audio_frame, "dtype", None) != _np.int16:
        scaled = _np.clip(audio_frame, -1.0, 1.0)
        audio_frame = (scaled * _np.iinfo(_np.int16).max).astype(_np.int16)

    predictions = model.predict(audio_frame)
    threshold = threshold if threshold is not None else settings.wakeword_threshold
    return any(score >= threshold for score in predictions.values())


__all__ = ["load_wakeword_model", "detect_wakeword"]
