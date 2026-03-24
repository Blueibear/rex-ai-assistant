"""Embedding helpers for custom wake words."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import numpy as np
else:
    try:  # pragma: no cover - optional dependency
        import numpy as np
    except Exception:  # pragma: no cover - optional dependency
        np = None

try:  # pragma: no cover - optional dependency
    import torch as _torch
except Exception:  # pragma: no cover - optional dependency
    _torch = None

torch: ModuleType | None = _torch

DEFAULT_EMBEDDING_BINS = 128


def compute_embedding(audio_frame: np.ndarray, *, bins: int = DEFAULT_EMBEDDING_BINS) -> np.ndarray:
    if np is None:
        raise RuntimeError("numpy is required for embedding computation")
    audio = np.asarray(audio_frame, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return np.zeros(bins, dtype=np.float32)

    spectrum = np.abs(np.fft.rfft(audio))
    if spectrum.size == 0:
        return np.zeros(bins, dtype=np.float32)

    spectrum = np.log1p(spectrum)
    x_old = np.linspace(0, 1, spectrum.size, endpoint=True)
    x_new = np.linspace(0, 1, bins, endpoint=True)
    resized = np.interp(x_new, x_old, spectrum).astype(np.float32)

    norm = np.linalg.norm(resized)
    if norm > 0:
        resized = resized / norm
    return cast("np.ndarray", resized)


def load_embedding(path: str | Path) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is required to load embedding files")
    if np is None:
        raise RuntimeError("numpy is required to load embedding files")

    data: Any = torch.load(Path(path), map_location="cpu")
    if isinstance(data, dict) and "embedding" in data:
        data = data["embedding"]

    if hasattr(data, "detach"):
        data = data.detach()

    if hasattr(data, "cpu"):
        data = data.cpu()

    if hasattr(data, "numpy"):
        array = data.numpy()
    else:
        array = np.asarray(data)

    array = np.asarray(array, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(array)
    if norm > 0:
        array = array / norm
    return cast("np.ndarray", array)


def save_embedding(path: str | Path, embedding: np.ndarray) -> None:
    if torch is None:
        raise RuntimeError("torch is required to save embedding files")
    if np is None:
        raise RuntimeError("numpy is required to save embedding files")
    torch.save(embedding, Path(path))
