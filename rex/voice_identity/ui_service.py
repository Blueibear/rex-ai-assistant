"""Voice identity helpers for GUI-driven enrollment flows.

This module keeps Electron/bridge code thin by centralising enrollment,
listing, and deletion behavior behind a small Python API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rex.config import load_config
from rex.config_manager import load_config as load_json_config
from rex.voice_identity.embeddings_store import EmbeddingsStore
from rex.voice_identity.enrollment import enroll_user
from rex.voice_identity.optional_deps import get_embedding_backend

_DEFAULT_MEMORY_DIR = Path(__file__).resolve().parents[2] / "Memory"
_NPY_FILENAME = "voice_embedding.npy"


def get_active_user_id() -> str:
    """Return the active runtime user ID for UI enrollment flows."""
    config = load_config()
    active_user = getattr(config, "default_user", None) or getattr(config, "user_id", None)
    return str(active_user or "default")


def list_enrollments(*, base_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """Return enrolled users and metadata for the UI list."""
    store = EmbeddingsStore(_resolve_base_dir(base_dir))
    records: list[dict[str, Any]] = []
    for user_id in store.list_enrolled_users():
        embedding = store.load(user_id)
        if embedding is None:
            continue
        records.append(
            {
                "user_id": user_id,
                "sample_count": int(embedding.sample_count),
                "updated_at": embedding.updated_at,
                "model_id": embedding.model_id,
            }
        )
    return records


def enroll_from_samples(
    user_id: str,
    audio_samples: list[list[float]],
    *,
    base_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Enroll *user_id* from JSON-serialisable float sample arrays."""
    samples = [np.asarray(sample, dtype=np.float32) for sample in audio_samples]
    if any(sample.size == 0 for sample in samples):
        raise ValueError("Enrollment samples must not be empty")

    voice_cfg = _load_voice_identity_config()
    backend = get_embedding_backend(
        str(voice_cfg.get("model_id", "synthetic")),
        dim=int(voice_cfg.get("embedding_dim", 192)),
    )
    resolved_base_dir = _resolve_base_dir(base_dir)
    enroll_user(user_id, samples, base_dir=resolved_base_dir, backend=backend)

    store = EmbeddingsStore(resolved_base_dir)
    embedding = store.load(user_id)
    if embedding is None:
        raise RuntimeError(f"Enrollment succeeded but no embedding was stored for {user_id!r}")

    return {
        "user_id": user_id,
        "sample_count": int(embedding.sample_count),
        "updated_at": embedding.updated_at,
        "model_id": embedding.model_id,
    }


def delete_enrollment(user_id: str, *, base_dir: Path | str | None = None) -> bool:
    """Delete enrollment artifacts for *user_id*."""
    resolved_base_dir = _resolve_base_dir(base_dir)
    store = EmbeddingsStore(resolved_base_dir)
    deleted_json = store.delete(user_id)
    user_dir = resolved_base_dir / user_id
    npy_path = user_dir / _NPY_FILENAME
    deleted_npy = False
    if npy_path.exists():
        npy_path.unlink()
        deleted_npy = True
    if user_dir.exists() and not any(user_dir.iterdir()):
        user_dir.rmdir()
    return deleted_json or deleted_npy


def _load_voice_identity_config() -> dict[str, Any]:
    raw_config = load_json_config()
    voice_identity = raw_config.get("voice_identity", {})
    if isinstance(voice_identity, dict):
        return voice_identity
    return {}


def _resolve_base_dir(base_dir: Path | str | None) -> Path:
    if base_dir is None:
        return _DEFAULT_MEMORY_DIR
    return Path(base_dir)
