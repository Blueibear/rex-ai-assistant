"""Helpers for loading wake-word models and evaluating predictions."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import openwakeword
    from openwakeword.model import Model as WakeWordModel
except Exception:  # pragma: no cover - openwakeword optional
    openwakeword = None  # type: ignore[assignment]
    WakeWordModel = object  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

DEFAULT_KEYWORD = os.getenv("REX_WAKEWORD_KEYWORD", "hey_jarvis")
DEFAULT_BACKEND = os.getenv("REX_WAKEWORD_BACKEND", "onnx")
_AVAILABLE_KEYWORDS = (
    {name.replace("_", " ") for name in getattr(openwakeword, "MODELS", {}).keys()}
    if openwakeword
    else set()
)


def _resolve_model_path(model_path: str | None = None) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if model_path:
        return Path(model_path)
    return repo_root / "rex.onnx"


def _normalise_keywords(keyword: str | None) -> list[str]:
    if not keyword:
        return [DEFAULT_KEYWORD.replace("_", " ")]
    return [part.strip().replace("_", " ") for part in keyword.split(",") if part.strip()]


def _ensure_builtin_keyword_resources(keywords: Iterable[str], backend: str) -> None:
    if openwakeword is None:
        return

    try:
        from openwakeword.utils import download_models
    except Exception as exc:  # pragma: no cover - defensive import guard
        logger.warning("Unable to prepare wake-word models automatically: %s", exc)
        return

    missing: list[str] = []
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
        download_models(model_names=list(dict.fromkeys(missing)))
        logger.info("Downloaded wake-word resources: %s", ", ".join(missing))
    except Exception as exc:  # pragma: no cover - network failures
        logger.warning("Failed to download wake-word models %s: %s", missing, exc)


def load_wakeword_model(
    *, keyword: str | None = None, model_path: str | None = None
) -> Tuple[WakeWordModel, str]:
    if openwakeword is None:
        raise RuntimeError("openwakeword is not installed")

    resolved_path = _resolve_model_path(model_path)
    requested_keywords = _normalise_keywords(keyword)

    if resolved_path.is_file() and resolved_path.stat().st_size > 0:
        models_to_load = [str(resolved_path)]
        active_label = resolved_path.stem
        print(f"[wakeword] Loading custom model from path: {resolved_path}")
    else:
        valid_keywords = [
            kw for kw in requested_keywords if not _AVAILABLE_KEYWORDS or kw in _AVAILABLE_KEYWORDS
        ]
        if not valid_keywords:
            fallback = DEFAULT_KEYWORD.replace("_", " ")
            print(
                f"[wakeword] Requested keywords {requested_keywords!r} not available, "
                f"falling back to default '{fallback}'"
            )
            valid_keywords = [fallback]

        models_to_load = valid_keywords
        active_label = ", ".join(valid_keywords)
        print(f"[wakeword] Loading built-in model for keyword: {active_label}")
        _ensure_builtin_keyword_resources(valid_keywords, DEFAULT_BACKEND)

    wake_model = WakeWordModel(
        wakeword_models=models_to_load,
        inference_framework=DEFAULT_BACKEND,
        enable_speex_noise_suppression=False,  # disabled for Windows compatibility
    )

    print(f"[wakeword] WakeWordModel initialized with models: {models_to_load}, backend={DEFAULT_BACKEND}")
    return wake_model, active_label


def detect_wakeword(
    model: WakeWordModel,
    audio_frame: np.ndarray,
    *,
    threshold: float = 0.5,
) -> bool:
    """Run wakeword detection on an audio frame with debug logging."""
    print(f"[wakeword] Received audio frame: shape={audio_frame.shape}, dtype={audio_frame.dtype}")

    if audio_frame.ndim != 1:
        audio_frame = np.reshape(audio_frame, (-1,))
        print(f"[wakeword] Reshaped audio_frame to shape={audio_frame.shape}")

    if audio_frame.dtype != np.int16:
        scaled = np.clip(audio_frame, -1.0, 1.0)
        audio_frame = (scaled * np.iinfo(np.int16).max).astype(np.int16)
        print("[wakeword] Converted audio_frame to int16 format")

    predictions = model.predict(audio_frame)
    print(f"[wakeword] Predictions: {predictions}")

    triggered = any(score >= threshold for score in predictions.values())
    if triggered:
        print(f"[wakeword] Wakeword detected! (threshold={threshold})")
    else:
        print(f"[wakeword] No wakeword detected (threshold={threshold})")

    return triggered


__all__ = ["load_wakeword_model", "detect_wakeword"]

