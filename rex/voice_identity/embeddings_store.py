"""Versioned per-user embeddings store backed by JSON files.

Storage layout::

    <base_dir>/<user_id>/voice_embeddings.json

Each file contains a JSON object with:

- ``model_id``: the embedding model that produced the vectors
- ``sample_count``: number of audio samples aggregated
- ``updated_at``: ISO-8601 timestamp of the last write
- ``embedding``: list of floats (the averaged speaker vector)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from rex.voice_identity.types import VoiceEmbedding

logger = logging.getLogger(__name__)

_FILENAME = "voice_embeddings.json"


class EmbeddingsStore:
    """Read and write per-user voice embeddings to disk.

    Parameters:
        base_dir: Root directory containing per-user subdirectories.
            Typically ``Memory/`` in the repository, or a ``tmp_path``
            during tests.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = Path(base_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, user_id: str) -> VoiceEmbedding | None:
        """Load the stored embedding for *user_id*, or ``None``."""
        path = self._path_for(user_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return VoiceEmbedding(
                vector=data["embedding"],
                model_id=data.get("model_id", "unknown"),
                sample_count=data.get("sample_count", 1),
                updated_at=data.get("updated_at", ""),
            )
        except Exception as exc:
            logger.warning("Failed to load embeddings for %s: %s", user_id, exc)
            return None

    def save(self, user_id: str, embedding: VoiceEmbedding) -> None:
        """Persist *embedding* for *user_id*, creating directories as needed."""
        path = self._path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_id": embedding.model_id,
            "sample_count": embedding.sample_count,
            "updated_at": embedding.updated_at or datetime.now(timezone.utc).isoformat(),
            "embedding": embedding.vector,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def delete(self, user_id: str) -> bool:
        """Remove the stored embedding for *user_id*. Returns ``True`` if deleted."""
        path = self._path_for(user_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_enrolled_users(self) -> list[str]:
        """Return user IDs that have stored embeddings."""
        if not self._base_dir.is_dir():
            return []
        users: list[str] = []
        for entry in sorted(self._base_dir.iterdir()):
            if entry.is_dir() and (entry / _FILENAME).exists():
                users.append(entry.name)
        return users

    def load_all(self) -> dict[str, VoiceEmbedding]:
        """Load embeddings for every enrolled user."""
        result: dict[str, VoiceEmbedding] = {}
        for user_id in self.list_enrolled_users():
            emb = self.load(user_id)
            if emb is not None:
                result[user_id] = emb
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _path_for(self, user_id: str) -> Path:
        """Return the JSON file path for a given user."""
        return self._base_dir / user_id / _FILENAME
