"""Helpers for loading wake-word models and evaluating predictions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import openwakeword
from openwakeword.model import Model as WakeWordModel

DEFAULT_KEYWORD = os.getenv("REX_WAKEWORD_KEYWORD", "hey_jarvis")
DEFAULT_BACKEND = os.getenv("REX_WAKEWORD_BACKEND", "onnx")
_AVAILABLE_KEYWORDS = {name.replace("_", " ") for name in openwakeword.MODELS.keys()}


def _resolve_model_path(model_path: str | None = None) -> Path:
    repo_root = Path(__file__).resolve().parent
    if model_path:
        return Path(model_path)
    return repo_root / "rex.onnx"


def load_wakeword_model(
    *, keyword: str | None = None, model_path: str | None = None
) -> Tuple[WakeWordModel, str]:
    """Return a configured openWakeWord model and the active wake phrase."""

    resolved_path = _resolve_model_path(model_path)
    requested_keyword = (keyword or DEFAULT_KEYWORD).replace("_", " ")

    if resolved_path.is_file() and resolved_path.stat().st_size > 0:
        models_to_load = [str(resolved_path)]
        active_label = resolved_path.stem
    else:
        if requested_keyword not in _AVAILABLE_KEYWORDS:
            print(
                f"[wakeword] Keyword '{requested_keyword}' not available; "
                f"falling back to '{DEFAULT_KEYWORD}'."
            )
            requested_keyword = DEFAULT_KEYWORD.replace("_", " ")
        models_to_load = [requested_keyword]
        active_label = requested_keyword

    wake_model = WakeWordModel(
        wakeword_models=models_to_load,
        inference_framework=DEFAULT_BACKEND,
        enable_speex_noise_suppression=True,
    )

    return wake_model, active_label


def detect_wakeword(
    model: WakeWordModel,
    audio_frame: np.ndarray,
    *,
    threshold: float = 0.5,
) -> bool:
    """Return ``True`` when ``audio_frame`` contains the active wake word."""

    if audio_frame.ndim != 1:
        audio_frame = np.reshape(audio_frame, (-1,))

    if audio_frame.dtype != np.int16:
        scaled = np.clip(audio_frame, -1.0, 1.0)
        audio_frame = (scaled * np.iinfo(np.int16).max).astype(np.int16)

    predictions = model.predict(audio_frame)
    return any(score >= threshold for score in predictions.values())


__all__ = ["load_wakeword_model", "detect_wakeword"]
