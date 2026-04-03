"""Custom wake word trainer using embedding-based detection.

Computes FFT spectral embeddings from audio samples, averages positive
samples into a mean embedding, and saves to config/wake_words/{slug}/embedding.pt
for use with EmbeddingWakeWordModel.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .embedding import compute_embedding, save_embedding

_CONFIG_DIR_DEFAULT = Path(__file__).resolve().parent.parent.parent / "config" / "wake_words"

MIN_POSITIVE_SAMPLES = 3
MIN_NEGATIVE_SAMPLES = 0  # negatives are optional for embedding approach


def _slugify(phrase: str) -> str:
    """Convert phrase to a filesystem-safe slug."""
    slug = phrase.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "_", slug)
    return slug or "custom_wake_word"


def train_from_samples(
    phrase: str,
    positive_samples: list[list[float]],
    negative_samples: list[list[float]],
    *,
    config_dir: Path | None = None,
) -> dict:
    """Train a custom wake word embedding from audio samples.

    Args:
        phrase: Human-readable wake word phrase (e.g. "hey rex").
        positive_samples: List of audio frames (16kHz float32 PCM) that contain
            the wake word.
        negative_samples: List of audio frames without the wake word (unused in
            embedding approach but accepted for API compatibility).
        config_dir: Override for config/wake_words directory (for testing).

    Returns:
        dict with keys: ok (bool), model_path (str), phrase (str).
        On error: ok=False, error (str).
    """
    if not phrase or not phrase.strip():
        return {"ok": False, "error": "Phrase must not be empty."}

    if len(positive_samples) < MIN_POSITIVE_SAMPLES:
        return {
            "ok": False,
            "error": f"Need at least {MIN_POSITIVE_SAMPLES} positive samples, got {len(positive_samples)}.",
        }

    target_dir = (config_dir or _CONFIG_DIR_DEFAULT) / _slugify(phrase)
    target_dir.mkdir(parents=True, exist_ok=True)

    embeddings: list[np.ndarray] = []
    for sample in positive_samples:
        frame = np.array(sample, dtype=np.float32)
        emb = compute_embedding(frame)
        embeddings.append(emb)

    mean_embedding = np.mean(np.stack(embeddings, axis=0), axis=0)

    embedding_path = target_dir / "embedding.pt"
    save_embedding(embedding_path, mean_embedding)

    # Save phrase metadata alongside the embedding for display purposes.
    meta_path = target_dir / "phrase.txt"
    meta_path.write_text(phrase.strip(), encoding="utf-8")

    return {
        "ok": True,
        "model_path": str(embedding_path),
        "phrase": phrase.strip(),
    }


def list_custom_wake_words(config_dir: Path | None = None) -> list[dict]:
    """Scan config/wake_words/ for trained custom embeddings.

    Returns a list of dicts with: id, name, engine, model_path.
    """
    base = config_dir or _CONFIG_DIR_DEFAULT
    if not base.is_dir():
        return []

    results: list[dict] = []
    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        embedding_file = subdir / "embedding.pt"
        if not embedding_file.is_file():
            continue

        phrase_file = subdir / "phrase.txt"
        if phrase_file.is_file():
            display = phrase_file.read_text(encoding="utf-8").strip()
        else:
            display = subdir.name.replace("_", " ").title()

        results.append(
            {
                "id": subdir.name,
                "name": display,
                "engine": "custom_embedding",
                "model_path": str(embedding_file),
            }
        )

    return results


__all__ = ["train_from_samples", "list_custom_wake_words"]
