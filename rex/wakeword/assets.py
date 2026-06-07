"""Helpers for installing and creating custom wake-word assets."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from shutil import copy2

from .utils import (
    WakeWordModelSelection,
    load_wakeword_model_with_metadata,
    resolve_custom_wakeword_embedding_path,
    resolve_custom_wakeword_model_path,
)


def _write_phrase_metadata(target: Path, phrase: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    (target.parent / "phrase.txt").write_text(phrase.strip(), encoding="utf-8")


def _copy_optional_sample(sample_path: str | None, target_dir: Path) -> None:
    if not sample_path:
        return
    sample = Path(sample_path)
    if not sample.is_file():
        raise FileNotFoundError(f"Sample file not found: {sample}")
    copy2(sample, target_dir / "sample.wav")


def install_custom_wakeword_asset(
    *,
    backend: str,
    phrase: str,
    source_path: str,
    target_path: str | None = None,
    sample_path: str | None = None,
) -> dict[str, object]:
    """Copy a supplied wake asset into the repo convention path and validate it."""
    normalized_backend = (backend or "").strip().lower()
    if normalized_backend not in {"custom_onnx", "custom_embedding"}:
        raise ValueError("backend must be 'custom_onnx' or 'custom_embedding'")
    if not phrase or not phrase.strip():
        raise ValueError("phrase must not be empty")

    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"Source asset not found: {source}")

    if normalized_backend == "custom_onnx":
        target = resolve_custom_wakeword_model_path(phrase, target_path)
        model_path = str(target)
        embedding_path = None
    else:
        target = resolve_custom_wakeword_embedding_path(phrase, target_path)
        model_path = None
        embedding_path = str(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    copy2(source, target)
    _write_phrase_metadata(target, phrase)
    _copy_optional_sample(sample_path, target.parent)

    _, selection = load_wakeword_model_with_metadata(
        keyword=phrase,
        model_path=model_path,
        embedding_path=embedding_path,
        backend=normalized_backend,
        fallback_to_builtin=False,
    )

    return {
        "ok": True,
        "backend": normalized_backend,
        "phrase": phrase.strip(),
        "asset_path": str(target),
        "selection": asdict(selection),
    }


def create_openwakeword_model_asset(
    *,
    phrase: str,
    target_path: str | None = None,
) -> dict[str, object]:
    """Record and validate a custom ONNX wake model using openWakeWord itself."""
    if not phrase or not phrase.strip():
        raise ValueError("phrase must not be empty")

    target = resolve_custom_wakeword_model_path(phrase, target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        import openwakeword
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("openwakeword is not installed") from exc

    model_cls = getattr(openwakeword, "Model", None)
    if model_cls is None:
        model_ns = getattr(openwakeword, "model", None)
        model_cls = getattr(model_ns, "Model", None) if model_ns is not None else None
    if model_cls is None:
        raise RuntimeError("openwakeword.Model is unavailable")

    recorder = model_cls(backend="onnx")
    record_wakeword = getattr(recorder, "record_wakeword", None)
    if not callable(record_wakeword):
        raise RuntimeError("Installed openwakeword does not support record_wakeword")

    record_wakeword(phrase.strip(), str(target))
    _write_phrase_metadata(target, phrase)

    _, selection = load_wakeword_model_with_metadata(
        keyword=phrase,
        model_path=str(target),
        backend="custom_onnx",
        fallback_to_builtin=False,
    )

    return {
        "ok": True,
        "backend": "custom_onnx",
        "phrase": phrase.strip(),
        "asset_path": str(target),
        "selection": asdict(selection),
    }


def validate_custom_wakeword_asset(
    *,
    backend: str,
    phrase: str,
    model_path: str | None = None,
    embedding_path: str | None = None,
) -> WakeWordModelSelection:
    """Load a custom asset without fallback so callers can validate it truthfully."""
    _, selection = load_wakeword_model_with_metadata(
        keyword=phrase,
        model_path=model_path,
        embedding_path=embedding_path,
        backend=backend,
        fallback_to_builtin=False,
    )
    return selection
