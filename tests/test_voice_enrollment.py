"""Unit tests for rex.voice_identity.enrollment.enroll_user()."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rex.voice_identity.embeddings_store import EmbeddingsStore
from rex.voice_identity.enrollment import _MIN_SAMPLES, _NPY_FILENAME, enroll_user

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_samples(n: int, length: int = 1600) -> list[np.ndarray]:
    """Return *n* distinct float32 audio arrays of *length* samples."""
    rng = np.random.default_rng(seed=42)
    return [rng.random(length, dtype=np.float32) for _ in range(n)]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_successful_enrollment_creates_json(tmp_path):
    """Enrolling a user creates the JSON embedding file."""
    samples = _make_samples(3)
    enroll_user("alice", samples, base_dir=tmp_path)

    store = EmbeddingsStore(tmp_path)
    emb = store.load("alice")
    assert emb is not None, "Expected embedding to be stored in JSON"
    assert emb.sample_count == 3
    assert len(emb.vector) > 0


def test_successful_enrollment_creates_npy(tmp_path):
    """Enrolling a user creates the .npy embedding file."""
    samples = _make_samples(3)
    enroll_user("bob", samples, base_dir=tmp_path)

    npy_path = tmp_path / "bob" / _NPY_FILENAME
    assert npy_path.exists(), f"Expected {npy_path} to exist after enrollment"
    arr = np.load(str(npy_path))
    assert arr.ndim == 1
    assert len(arr) > 0


def test_npy_matches_json_embedding(tmp_path):
    """The .npy vector and JSON vector are the same (within float tolerance)."""
    samples = _make_samples(4)
    enroll_user("carol", samples, base_dir=tmp_path)

    store = EmbeddingsStore(tmp_path)
    emb = store.load("carol")
    assert emb is not None

    npy_path = tmp_path / "carol" / _NPY_FILENAME
    arr = np.load(str(npy_path))

    assert len(emb.vector) == len(arr)
    for json_val, npy_val in zip(emb.vector, arr.tolist(), strict=False):
        assert abs(json_val - npy_val) < 1e-5, "JSON and .npy vectors differ"


def test_embedding_is_unit_length(tmp_path):
    """The stored embedding vector is normalised to unit length."""
    samples = _make_samples(3)
    enroll_user("dave", samples, base_dir=tmp_path)

    store = EmbeddingsStore(tmp_path)
    emb = store.load("dave")
    assert emb is not None

    magnitude = math.sqrt(sum(v * v for v in emb.vector))
    assert abs(magnitude - 1.0) < 1e-5, f"Expected unit vector; got magnitude {magnitude}"


def test_insufficient_samples_raises_value_error(tmp_path):
    """Fewer than 3 samples raises ValueError."""
    samples = _make_samples(2)
    with pytest.raises(ValueError, match="at least"):
        enroll_user("eve", samples, base_dir=tmp_path)


def test_zero_samples_raises_value_error(tmp_path):
    """Empty sample list raises ValueError."""
    with pytest.raises(ValueError, match="at least"):
        enroll_user("frank", [], base_dir=tmp_path)


def test_exactly_min_samples_accepted(tmp_path):
    """Exactly _MIN_SAMPLES samples are accepted without error."""
    samples = _make_samples(_MIN_SAMPLES)
    enroll_user("grace", samples, base_dir=tmp_path)  # should not raise

    store = EmbeddingsStore(tmp_path)
    emb = store.load("grace")
    assert emb is not None
    assert emb.sample_count == _MIN_SAMPLES


def test_sample_count_stored_correctly(tmp_path):
    """sample_count in JSON reflects the number of input samples."""
    n = 5
    samples = _make_samples(n)
    enroll_user("harry", samples, base_dir=tmp_path)

    store = EmbeddingsStore(tmp_path)
    emb = store.load("harry")
    assert emb is not None
    assert emb.sample_count == n


def test_enrollment_does_not_affect_other_users(tmp_path):
    """Enrolling user A does not create files for user B."""
    samples = _make_samples(3)
    enroll_user("irene", samples, base_dir=tmp_path)

    assert not (tmp_path / "other_user").exists()


def test_default_backend_is_synthetic(tmp_path):
    """When no backend is supplied, the synthetic backend is used."""
    samples = _make_samples(3)
    enroll_user("judy", samples, base_dir=tmp_path)

    store = EmbeddingsStore(tmp_path)
    emb = store.load("judy")
    assert emb is not None
    assert emb.model_id == "synthetic"
