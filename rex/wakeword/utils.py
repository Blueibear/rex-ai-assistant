"""Helpers for loading wake-word models and evaluating predictions."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path


def _import_optional(module_name: str):
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    if find_spec(module_name) is None:
        return None
    return import_module(module_name)


np = _import_optional("numpy")

def _lazy_import_openwakeword():
    module = _import_optional("openwakeword")
    if module is None:
        return None, object
    model_cls = getattr(module, "model", None)
    model_type = getattr(model_cls, "Model", object) if model_cls else object
    return module, model_type


WakeWordModel = object
openwakeword = None
_OPENWAKEWORD_MODULE = None
_WAKEWORD_MODEL = object


def _get_openwakeword():
    global _OPENWAKEWORD_MODULE, _WAKEWORD_MODEL, openwakeword
    if openwakeword is not None and openwakeword is not _OPENWAKEWORD_MODULE:
        _OPENWAKEWORD_MODULE = openwakeword
        _WAKEWORD_MODEL = WakeWordModel if WakeWordModel is not object else object
        return _OPENWAKEWORD_MODULE, _WAKEWORD_MODEL

    module = sys.modules.get("openwakeword")
    if module is not None and module is not _OPENWAKEWORD_MODULE:
        model_cls = getattr(module, "model", None)
        _OPENWAKEWORD_MODULE = module
        _WAKEWORD_MODEL = getattr(model_cls, "Model", object) if model_cls else object
        openwakeword = module
        return _OPENWAKEWORD_MODULE, _WAKEWORD_MODEL

    if _OPENWAKEWORD_MODULE is None:
        _OPENWAKEWORD_MODULE, _WAKEWORD_MODEL = _lazy_import_openwakeword()
        openwakeword = _OPENWAKEWORD_MODULE
    return _OPENWAKEWORD_MODULE, _WAKEWORD_MODEL

from .embedding import compute_embedding, load_embedding  # noqa: E402
from .selection import (  # noqa: E402
    list_openwakeword_keywords,
    normalize_keyword,
    select_fallback_keyword,
    split_keywords,
)

logger = logging.getLogger(__name__)

# Get defaults from config (with fallbacks)
try:
    from rex.config import settings

    DEFAULT_BACKEND = getattr(settings, "wakeword_backend", "openwakeword")
    DEFAULT_FALLBACK_KEYWORD = getattr(settings, "wakeword_fallback_keyword", "hey jarvis")
    DEFAULT_FALLBACK_TO_BUILTIN = bool(getattr(settings, "wakeword_fallback_to_builtin", True))
except Exception:
    DEFAULT_BACKEND = "openwakeword"
    DEFAULT_FALLBACK_KEYWORD = "hey jarvis"
    DEFAULT_FALLBACK_TO_BUILTIN = True


def _resolve_model_path(model_path: str | None = None) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if model_path:
        return Path(model_path)
    return repo_root / "rex.onnx"


def _normalize_backend(backend: str | None) -> tuple[str, str]:
    resolved = (backend or DEFAULT_BACKEND or "openwakeword").strip().lower()
    if resolved in {"onnx", "tflite"}:
        return "openwakeword", resolved
    return resolved, "onnx"


def _ensure_builtin_keyword_resources(keywords: Iterable[str], backend: str) -> None:
    openwakeword, _ = _get_openwakeword()
    if openwakeword is None:
        return

    try:
        from openwakeword.utils import download_models
    except Exception as exc:
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
    except Exception as exc:
        logger.warning("Failed to download wake-word models %s: %s", missing, exc)


class EmbeddingWakeWordModel:
    """Wake word model that compares embeddings with cosine similarity."""

    def __init__(self, embedding: np.ndarray, *, label: str) -> None:
        self._embedding = embedding
        self._label = label

    def predict(self, audio_frame: np.ndarray) -> dict[str, float]:
        if np is None:
            raise RuntimeError("numpy is required for embedding wake word detection")
        candidate = compute_embedding(audio_frame, bins=self._embedding.size)
        score = float(np.dot(candidate, self._embedding))
        return {self._label: score}


def _load_builtin_openwakeword_model(
    *,
    keyword: str | None,
    inference_framework: str,
    fallback_keyword: str | None,
) -> tuple[WakeWordModel, str]:
    openwakeword, model_cls = _get_openwakeword()
    if openwakeword is None:
        raise RuntimeError("openwakeword is not installed")
    if model_cls is object and WakeWordModel is not object:
        model_cls = WakeWordModel

    requested_keywords = split_keywords(keyword)
    available = list_openwakeword_keywords(openwakeword)
    if not available:
        raise RuntimeError("No openwakeword models available.")

    available_map = {normalize_keyword(item): item for item in available}
    valid_keywords = [
        available_map[normalize_keyword(item)]
        for item in requested_keywords
        if normalize_keyword(item) in available_map
    ]

    if not valid_keywords:
        fallback = select_fallback_keyword(
            available,
            fallback_keyword=fallback_keyword,
        )
        logger.warning(
            "Requested wake word %s not available. Falling back to '%s'. Available models: %s",
            requested_keywords or [keyword or ""],
            fallback,
            available,
        )
        valid_keywords = [fallback]

    models_to_load: list[str] = valid_keywords
    active_label = ", ".join(valid_keywords)
    _ensure_builtin_keyword_resources(valid_keywords, inference_framework)

    wake_model = model_cls(
        wakeword_models=models_to_load,
        inference_framework=inference_framework,
        enable_speex_noise_suppression=False,  # disabled for Windows compatibility
    )
    return wake_model, active_label


def load_wakeword_model(
    *,
    keyword: str | None = None,
    model_path: str | None = None,
    embedding_path: str | None = None,
    backend: str | None = None,
    fallback_to_builtin: bool | None = None,
    fallback_keyword: str | None = None,
) -> tuple[WakeWordModel | EmbeddingWakeWordModel, str]:
    resolved_backend, inference_framework = _normalize_backend(backend)
    fallback_to_builtin = (
        DEFAULT_FALLBACK_TO_BUILTIN if fallback_to_builtin is None else bool(fallback_to_builtin)
    )
    fallback_keyword = fallback_keyword or DEFAULT_FALLBACK_KEYWORD

    if resolved_backend == "custom_onnx":
        resolved_path = Path(model_path) if model_path else _resolve_model_path(None)
        if resolved_path.is_file() and resolved_path.stat().st_size > 0:
            openwakeword, model_cls = _get_openwakeword()
            if openwakeword is None:
                raise RuntimeError("openwakeword is not installed")
            wake_model = model_cls(
                wakeword_models=[str(resolved_path)],
                inference_framework="onnx",
                enable_speex_noise_suppression=False,
            )
            return wake_model, resolved_path.stem
        if not fallback_to_builtin:
            raise RuntimeError(f"Custom wake word model not found at {resolved_path}")
        logger.warning(
            "Custom wake word model not found at %s. Falling back to built-in keyword '%s'.",
            resolved_path,
            fallback_keyword,
        )
        return _load_builtin_openwakeword_model(
            keyword=fallback_keyword,
            inference_framework=inference_framework,
            fallback_keyword=fallback_keyword,
        )

    if resolved_backend == "custom_embedding":
        resolved_path = Path(embedding_path) if embedding_path else Path()
        if resolved_path.is_file() and resolved_path.stat().st_size > 0:
            embedding = load_embedding(resolved_path)
            return EmbeddingWakeWordModel(embedding, label=resolved_path.stem), resolved_path.stem
        if not fallback_to_builtin:
            raise RuntimeError(f"Custom wake word embedding not found at {resolved_path}")
        logger.warning(
            "Custom wake word embedding not found at %s. Falling back to built-in keyword '%s'.",
            resolved_path,
            fallback_keyword,
        )
        return _load_builtin_openwakeword_model(
            keyword=fallback_keyword,
            inference_framework=inference_framework,
            fallback_keyword=fallback_keyword,
        )

    resolved_keyword = keyword
    return _load_builtin_openwakeword_model(
        keyword=resolved_keyword,
        inference_framework=inference_framework,
        fallback_keyword=fallback_keyword,
    )


def detect_wakeword(
    model: WakeWordModel,
    audio_frame: np.ndarray,
    *,
    threshold: float = 0.5,
) -> bool:
    """Run wakeword detection on an audio frame with debug logging."""
    if np is None:
        raise RuntimeError("numpy is required for wake word detection")
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
