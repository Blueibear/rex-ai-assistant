"""Wake-word detection utilities for Rex."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence, Tuple

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


def _normalise_keywords(keyword: str | Sequence[str] | None) -> List[str]:
    if keyword is None:
        return [DEFAULT_KEYWORD.replace("_", " ")]

    if isinstance(keyword, str):
        raw_items = [item.strip() for item in keyword.split(",")]
    else:
        raw_items = []
        for item in keyword:
            if isinstance(item, str):
                raw_items.extend(part.strip() for part in item.split(","))

    cleaned = [item.replace("_", " ") for item in raw_items if item]
    return cleaned or [DEFAULT_KEYWORD.replace("_", " ")]


def load_wakeword_model(
    *, keyword: str | Sequence[str] | None = None, model_path: str | None = None
) -> Tuple[WakeWordModel, str]:
    """Return a configured openWakeWord model and the active wake phrase."""

    resolved_path = _resolve_model_path(model_path)
    requested_keywords = _normalise_keywords(keyword)

    if resolved_path.is_file() and resolved_path.stat().st_size > 0:
        models_to_load: List[str | os.PathLike[str]] = [str(resolved_path)]
        active_label = resolved_path.stem
    else:
        valid_keywords = [kw for kw in requested_keywords if kw in _AVAILABLE_KEYWORDS]
        if not valid_keywords:
            fallback = DEFAULT_KEYWORD.replace("_", " ")
            print(
                f"[wakeword] Keywords {requested_keywords!r} unavailable; "
                f"falling back to '{fallback}'."
            )
            valid_keywords = [fallback]
        models_to_load = valid_keywords
        active_label = ", ".join(valid_keywords)

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

