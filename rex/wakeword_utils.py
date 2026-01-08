"""Wake-word detection utilities for Rex."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import openwakeword
from openwakeword.model import Model as WakeWordModel

logger = logging.getLogger(__name__)

DEFAULT_KEYWORD = os.getenv("REX_WAKEWORD_KEYWORD", "hey_jarvis")
DEFAULT_BACKEND = os.getenv("REX_WAKEWORD_BACKEND", "onnx")
_AVAILABLE_KEYWORDS = {name.replace("_", " ") for name in openwakeword.MODELS.keys()}


def _resolve_model_path(model_path: str | None = None) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
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


def _ensure_builtin_keyword_resources(keywords: Sequence[str], backend: str) -> None:
    """Ensure packaged wake-word models are downloaded before loading."""

    try:
        from openwakeword.utils import download_models
    except Exception as exc:  # pragma: no cover - defensive import guard
        logger.warning("Unable to import openwakeword.utils.download_models: %s", exc)
        return

    missing: List[str] = []
    backend = (backend or "").lower()

    for keyword in keywords:
        key = keyword.replace(" ", "_")
        info = openwakeword.MODELS.get(key)
        if not info:
            continue

        model_path = Path(info["model_path"])
        if backend == "onnx":
            model_path = model_path.with_suffix(".onnx")

        if model_path.exists():
            continue

        download_name = info["download_url"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        missing.append(download_name)

    if not missing:
        return

    try:
        # dict.fromkeys preserves order while dropping duplicates
        download_models(model_names=list(dict.fromkeys(missing)))
    except Exception as exc:  # pragma: no cover - network failures
        logger.warning("Failed to download wake-word models %s: %s", missing, exc)


def load_wakeword_model(
    *, keyword: str | Sequence[str] | None = None, model_path: str | None = None
) -> Tuple[WakeWordModel, str]:
    """Return a configured openWakeWord model and the active wake phrase."""

    resolved_path = _resolve_model_path(model_path)
    requested_keywords = _normalise_keywords(keyword)

    logger.info(f"Loading wake word model - requested keywords: {requested_keywords}")
    logger.info(f"Available openwakeword models: {sorted(_AVAILABLE_KEYWORDS)}")

    if resolved_path.is_file() and resolved_path.stat().st_size > 0:
        models_to_load: List[str | os.PathLike[str]] = [str(resolved_path)]
        active_label = resolved_path.stem
        logger.info(f"Using custom model file: {resolved_path}")
    else:
        valid_keywords = [kw for kw in requested_keywords if kw in _AVAILABLE_KEYWORDS]
        if not valid_keywords:
            fallback = DEFAULT_KEYWORD.replace("_", " ")
            logger.warning(
                f"Keywords {requested_keywords!r} not available in openwakeword models. "
                f"Falling back to '{fallback}'. "
                f"Available models: {sorted(_AVAILABLE_KEYWORDS)}"
            )
            print(f"[WAKEWORD] Keywords {requested_keywords!r} unavailable; falling back to '{fallback}'.")
            valid_keywords = [fallback]
        models_to_load = valid_keywords
        active_label = ", ".join(valid_keywords)
        logger.info(f"Loading built-in models: {models_to_load}")
        _ensure_builtin_keyword_resources(valid_keywords, DEFAULT_BACKEND)

    wake_model = WakeWordModel(
        wakeword_models=models_to_load,
        inference_framework=DEFAULT_BACKEND,
        enable_speex_noise_suppression=False,  # Disabled for Windows compatibility (speexdsp_ns not available)
    )

    logger.info(f"Wake word model loaded successfully. Active wake phrase: '{active_label}'")
    print(f"[WAKEWORD] Model loaded. Say: '{active_label}' (threshold: adjust in .env if needed)")
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

    # TEMPORARY DEBUG: Log EVERY detection to diagnose issue
    scores_str = ", ".join(f"{k}: {v:.3f}" for k, v in predictions.items())
    logger.info(f"Wake word scores: {scores_str} (threshold: {threshold})")

    detected = any(score >= threshold for score in predictions.values())
    if detected:
        logger.info(f"✓✓✓ WAKE WORD DETECTED! Scores: {scores_str} ✓✓✓")

    return detected


__all__ = ["load_wakeword_model", "detect_wakeword"]
