"""Helpers for loading wake-word models and evaluating predictions."""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
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
    DEFAULT_AUTO_GAIN = bool(getattr(settings, "wakeword_auto_gain", True))
    DEFAULT_TARGET_PEAK = float(getattr(settings, "wakeword_target_peak", 0.35))
    DEFAULT_MAX_GAIN = float(getattr(settings, "wakeword_max_gain", 30.0))
    DEFAULT_MIN_RMS_FOR_GAIN = float(getattr(settings, "wakeword_min_rms_for_gain", 0.0005))
except Exception:
    DEFAULT_BACKEND = "openwakeword"
    DEFAULT_FALLBACK_KEYWORD = "hey jarvis"
    DEFAULT_FALLBACK_TO_BUILTIN = True
    DEFAULT_AUTO_GAIN = True
    DEFAULT_TARGET_PEAK = 0.35
    DEFAULT_MAX_GAIN = 30.0
    DEFAULT_MIN_RMS_FOR_GAIN = 0.0005


def _resolve_model_path(model_path: str | None = None) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if model_path:
        return Path(model_path)
    return repo_root / "rex.onnx"


def slugify_wakeword_phrase(phrase: str | None, *, default: str = "hey_rex") -> str:
    raw = (phrase or "").strip().lower()
    if not raw:
        return default
    slug = re.sub(r"[^\w\s-]", "", raw)
    slug = re.sub(r"[\s-]+", "_", slug).strip("_")
    return slug or default


def _custom_wakeword_assets_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "wake_words"


def resolve_custom_wakeword_model_path(
    phrase: str | None,
    model_path: str | None = None,
) -> Path:
    if model_path and str(model_path).strip():
        return Path(str(model_path).strip())
    return _custom_wakeword_assets_dir() / slugify_wakeword_phrase(phrase) / "model.onnx"


def resolve_custom_wakeword_embedding_path(
    phrase: str | None,
    embedding_path: str | None = None,
) -> Path:
    if embedding_path and str(embedding_path).strip():
        return Path(str(embedding_path).strip())
    return _custom_wakeword_assets_dir() / slugify_wakeword_phrase(phrase) / "embedding.pt"


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

    def __init__(self, embedding: np.ndarray, *, label: str) -> None:  # type: ignore[name-defined]
        self._embedding = embedding
        self._label = label

    def predict(self, audio_frame: np.ndarray) -> dict[str, float]:  # type: ignore[name-defined]
        if np is None:
            raise RuntimeError("numpy is required for embedding wake word detection")
        candidate = compute_embedding(audio_frame, bins=self._embedding.size)
        score = float(np.dot(candidate, self._embedding))
        return {self._label: score}


@dataclass(frozen=True)
class WakeWordDetectionResult:
    """Detailed result for one wake-word detection frame."""

    triggered: bool
    threshold: float
    predictions: dict[str, float]
    confidence: float
    keyword: str | None
    reason: str
    audio_rms: float = 0.0
    audio_peak: float = 0.0
    effective_peak: float = 0.0
    applied_gain: float = 1.0
    gain_limit: float = 1.0
    target_peak: float = 0.0


@dataclass(frozen=True)
class WakeWordModelSelection:
    """Resolved wake-word selection including fallback and asset details."""

    requested_backend: str
    active_backend: str
    requested_phrase: str | None
    active_label: str
    requested_model_path: str | None = None
    requested_embedding_path: str | None = None
    resolved_model_path: str | None = None
    resolved_embedding_path: str | None = None
    used_fallback: bool = False
    fallback_keyword: str | None = None
    validation_error: str | None = None


def _runtime_setting(name: str, default):
    try:
        from rex import config as config_module

        current_settings = getattr(config_module, "settings", None)
        if current_settings is not None:
            return getattr(current_settings, name, default)
    except Exception:
        pass

    try:
        return getattr(settings, name, default)
    except Exception:
        return default


def _audio_level(samples) -> tuple[float, float]:
    if np is None or samples.size == 0:
        return 0.0, 0.0
    abs_samples = np.abs(samples)
    return (
        float(np.sqrt(np.mean(samples * samples))),
        float(np.max(abs_samples)),
    )


def _prepare_float_audio_for_detection(
    audio_frame,
) -> tuple[object, float, float, float, float, float]:
    """Return int16 wake audio plus original level stats.

    The wake detector expects a useful int16 signal. Normal desktop mics can
    arrive with very low float amplitudes, which makes wake detection require
    near-mic shouting even when the phrase is audible to a person.
    """
    frame = np.asarray(audio_frame, dtype=np.float32).reshape(-1)
    frame = np.nan_to_num(frame, nan=0.0, posinf=1.0, neginf=-1.0)
    frame = np.clip(frame, -1.0, 1.0)
    audio_rms, audio_peak = _audio_level(frame)
    auto_gain = bool(_runtime_setting("wakeword_auto_gain", DEFAULT_AUTO_GAIN))
    target_peak = float(_runtime_setting("wakeword_target_peak", DEFAULT_TARGET_PEAK))
    max_gain = float(_runtime_setting("wakeword_max_gain", DEFAULT_MAX_GAIN))
    min_rms_for_gain = float(
        _runtime_setting("wakeword_min_rms_for_gain", DEFAULT_MIN_RMS_FOR_GAIN)
    )

    gain = 1.0
    if auto_gain and audio_rms >= min_rms_for_gain and 0.0 < audio_peak < target_peak:
        gain = min(max_gain, target_peak / audio_peak)
        frame = np.clip(frame * gain, -1.0, 1.0)

    pcm16 = (frame * np.iinfo(np.int16).max).astype(np.int16)
    return pcm16, audio_rms, audio_peak, gain, max_gain, target_peak


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
    model, selection = load_wakeword_model_with_metadata(
        keyword=keyword,
        model_path=model_path,
        embedding_path=embedding_path,
        backend=backend,
        fallback_to_builtin=fallback_to_builtin,
        fallback_keyword=fallback_keyword,
    )
    return model, selection.active_label


def load_wakeword_model_with_metadata(
    *,
    keyword: str | None = None,
    model_path: str | None = None,
    embedding_path: str | None = None,
    backend: str | None = None,
    fallback_to_builtin: bool | None = None,
    fallback_keyword: str | None = None,
) -> tuple[WakeWordModel | EmbeddingWakeWordModel, WakeWordModelSelection]:
    resolved_backend, inference_framework = _normalize_backend(backend)
    fallback_to_builtin = (
        DEFAULT_FALLBACK_TO_BUILTIN if fallback_to_builtin is None else bool(fallback_to_builtin)
    )
    fallback_keyword = fallback_keyword or DEFAULT_FALLBACK_KEYWORD

    if resolved_backend == "custom_onnx":
        resolved_path = resolve_custom_wakeword_model_path(keyword, model_path)
        if resolved_path.is_file() and resolved_path.stat().st_size > 0:
            try:
                openwakeword, model_cls = _get_openwakeword()
                if openwakeword is None:
                    raise RuntimeError("openwakeword is not installed")
                wake_model = model_cls(
                    wakeword_models=[str(resolved_path)],
                    inference_framework="onnx",
                    enable_speex_noise_suppression=False,
                )
                return wake_model, WakeWordModelSelection(
                    requested_backend=resolved_backend,
                    active_backend="custom_onnx",
                    requested_phrase=keyword,
                    active_label=resolved_path.stem,
                    requested_model_path=model_path,
                    resolved_model_path=str(resolved_path),
                    fallback_keyword=fallback_keyword,
                )
            except Exception as exc:
                if not fallback_to_builtin:
                    raise RuntimeError(
                        f"Custom wake word model at {resolved_path} failed validation/load: {exc}"
                    ) from exc
                logger.warning(
                    "Custom wake word model at %s failed validation/load (%s). "
                    "Falling back to built-in keyword '%s'.",
                    resolved_path,
                    exc,
                    fallback_keyword,
                )
                fallback_model, fallback_label = _load_builtin_openwakeword_model(
                    keyword=fallback_keyword,
                    inference_framework=inference_framework,
                    fallback_keyword=fallback_keyword,
                )
                return fallback_model, WakeWordModelSelection(
                    requested_backend=resolved_backend,
                    active_backend="openwakeword",
                    requested_phrase=keyword,
                    active_label=fallback_label,
                    requested_model_path=model_path,
                    resolved_model_path=str(resolved_path),
                    used_fallback=True,
                    fallback_keyword=fallback_keyword,
                    validation_error=str(exc),
                )
        if not fallback_to_builtin:
            raise RuntimeError(f"Custom wake word model not found at {resolved_path}")
        logger.warning(
            "Custom wake word model not found at %s. Falling back to built-in keyword '%s'.",
            resolved_path,
            fallback_keyword,
        )
        fallback_model, fallback_label = _load_builtin_openwakeword_model(
            keyword=fallback_keyword,
            inference_framework=inference_framework,
            fallback_keyword=fallback_keyword,
        )
        return fallback_model, WakeWordModelSelection(
            requested_backend=resolved_backend,
            active_backend="openwakeword",
            requested_phrase=keyword,
            active_label=fallback_label,
            requested_model_path=model_path,
            resolved_model_path=str(resolved_path),
            used_fallback=True,
            fallback_keyword=fallback_keyword,
        )

    if resolved_backend == "custom_embedding":
        resolved_path = resolve_custom_wakeword_embedding_path(keyword, embedding_path)
        if resolved_path.is_file() and resolved_path.stat().st_size > 0:
            try:
                embedding = load_embedding(resolved_path)
                return EmbeddingWakeWordModel(
                    embedding, label=resolved_path.stem
                ), WakeWordModelSelection(
                    requested_backend=resolved_backend,
                    active_backend="custom_embedding",
                    requested_phrase=keyword,
                    active_label=resolved_path.stem,
                    requested_embedding_path=embedding_path,
                    resolved_embedding_path=str(resolved_path),
                    fallback_keyword=fallback_keyword,
                )
            except Exception as exc:
                if not fallback_to_builtin:
                    raise RuntimeError(
                        f"Custom wake word embedding at {resolved_path} failed validation/load: {exc}"
                    ) from exc
                logger.warning(
                    "Custom wake word embedding at %s failed validation/load (%s). "
                    "Falling back to built-in keyword '%s'.",
                    resolved_path,
                    exc,
                    fallback_keyword,
                )
                fallback_model, fallback_label = _load_builtin_openwakeword_model(
                    keyword=fallback_keyword,
                    inference_framework=inference_framework,
                    fallback_keyword=fallback_keyword,
                )
                return fallback_model, WakeWordModelSelection(
                    requested_backend=resolved_backend,
                    active_backend="openwakeword",
                    requested_phrase=keyword,
                    active_label=fallback_label,
                    requested_embedding_path=embedding_path,
                    resolved_embedding_path=str(resolved_path),
                    used_fallback=True,
                    fallback_keyword=fallback_keyword,
                    validation_error=str(exc),
                )
        if not fallback_to_builtin:
            raise RuntimeError(f"Custom wake word embedding not found at {resolved_path}")
        logger.warning(
            "Custom wake word embedding not found at %s. Falling back to built-in keyword '%s'.",
            resolved_path,
            fallback_keyword,
        )
        fallback_model, fallback_label = _load_builtin_openwakeword_model(
            keyword=fallback_keyword,
            inference_framework=inference_framework,
            fallback_keyword=fallback_keyword,
        )
        return fallback_model, WakeWordModelSelection(
            requested_backend=resolved_backend,
            active_backend="openwakeword",
            requested_phrase=keyword,
            active_label=fallback_label,
            requested_embedding_path=embedding_path,
            resolved_embedding_path=str(resolved_path),
            used_fallback=True,
            fallback_keyword=fallback_keyword,
        )

    resolved_keyword = keyword
    wake_model, label = _load_builtin_openwakeword_model(
        keyword=resolved_keyword,
        inference_framework=inference_framework,
        fallback_keyword=fallback_keyword,
    )
    return wake_model, WakeWordModelSelection(
        requested_backend=resolved_backend,
        active_backend="openwakeword",
        requested_phrase=keyword,
        active_label=label,
        fallback_keyword=fallback_keyword,
    )


def detect_wakeword(
    model: WakeWordModel,
    audio_frame: np.ndarray,  # type: ignore[name-defined]
    *,
    threshold: float = 0.5,
) -> bool:
    """Run wakeword detection on an audio frame with debug logging."""
    return evaluate_wakeword(model, audio_frame, threshold=threshold).triggered


def evaluate_wakeword(
    model: WakeWordModel,
    audio_frame: np.ndarray,  # type: ignore[name-defined]
    *,
    threshold: float = 0.5,
) -> WakeWordDetectionResult:
    """Run wake-word detection and return score details for instrumentation."""
    if np is None:
        raise RuntimeError("numpy is required for wake word detection")
    logger.debug("Received audio frame: shape=%s, dtype=%s", audio_frame.shape, audio_frame.dtype)

    if audio_frame.ndim != 1:
        audio_frame = np.reshape(audio_frame, (-1,))
        logger.debug("Reshaped audio_frame to shape=%s", audio_frame.shape)

    applied_gain = 1.0
    audio_rms = 0.0
    audio_peak = 0.0
    gain_limit = 1.0
    target_peak = 0.0
    if audio_frame.dtype != np.int16:
        (
            audio_frame,
            audio_rms,
            audio_peak,
            applied_gain,
            gain_limit,
            target_peak,
        ) = _prepare_float_audio_for_detection(audio_frame)
        logger.debug(
            (
                "Converted audio_frame to int16 format "
                "(rms=%.6f peak=%.6f gain=%.2f max_gain=%.2f)"
            ),
            audio_rms,
            audio_peak,
            applied_gain,
            gain_limit,
        )
    else:
        int16_float = np.asarray(audio_frame, dtype=np.float32) / float(np.iinfo(np.int16).max)
        audio_rms, audio_peak = _audio_level(int16_float)

    predictions = model.predict(audio_frame)  # type: ignore[attr-defined]
    predictions = {str(key): float(value) for key, value in predictions.items()}
    logger.debug("Predictions: %s", predictions)

    keyword: str | None = None
    confidence = 0.0
    if predictions:
        keyword, confidence = max(predictions.items(), key=lambda item: item[1])

    triggered = any(score >= threshold for score in predictions.values())
    if triggered:
        logger.debug("Wake word detected (threshold=%.2f)", threshold)
        reason = "confidence_met_threshold"
    else:
        logger.debug("No wake word detected (threshold=%.2f)", threshold)
        reason = "below_threshold"

    return WakeWordDetectionResult(
        triggered=triggered,
        threshold=threshold,
        predictions=predictions,
        confidence=confidence,
        keyword=keyword,
        reason=reason,
        audio_rms=audio_rms,
        audio_peak=audio_peak,
        effective_peak=min(audio_peak * applied_gain, 1.0) if audio_peak > 0.0 else 0.0,
        applied_gain=applied_gain,
        gain_limit=gain_limit,
        target_peak=target_peak,
    )


__all__ = [
    "WakeWordDetectionResult",
    "WakeWordModelSelection",
    "detect_wakeword",
    "evaluate_wakeword",
    "load_wakeword_model",
    "load_wakeword_model_with_metadata",
    "resolve_custom_wakeword_embedding_path",
    "resolve_custom_wakeword_model_path",
    "slugify_wakeword_phrase",
]
