from __future__ import annotations

import json

import numpy as np

from rex.voice_identity.ui_service import delete_enrollment, enroll_from_samples, list_enrollments


def _sample(seed: int, length: int = 1600) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.random(length, dtype=np.float32).tolist()


def test_enroll_from_samples_persists_metadata(tmp_path):
    enrollment = enroll_from_samples(
        "alice",
        [_sample(1), _sample(2), _sample(3)],
        base_dir=tmp_path,
    )

    assert enrollment["user_id"] == "alice"
    assert enrollment["sample_count"] == 3
    assert enrollment["model_id"] == "synthetic"
    assert isinstance(enrollment["updated_at"], str)
    assert (tmp_path / "alice" / "voice_embeddings.json").exists()
    assert (tmp_path / "alice" / "voice_embedding.npy").exists()


def test_list_enrollments_returns_sorted_records(tmp_path):
    enroll_from_samples("bob", [_sample(4), _sample(5), _sample(6)], base_dir=tmp_path)
    enroll_from_samples("alice", [_sample(7), _sample(8), _sample(9)], base_dir=tmp_path)

    records = list_enrollments(base_dir=tmp_path)

    assert [record["user_id"] for record in records] == ["alice", "bob"]
    assert all(record["sample_count"] == 3 for record in records)


def test_delete_enrollment_removes_json_and_npy(tmp_path):
    enroll_from_samples("carol", [_sample(10), _sample(11), _sample(12)], base_dir=tmp_path)

    assert delete_enrollment("carol", base_dir=tmp_path) is True
    assert not (tmp_path / "carol" / "voice_embeddings.json").exists()
    assert not (tmp_path / "carol" / "voice_embedding.npy").exists()


def test_delete_enrollment_returns_false_when_missing(tmp_path):
    assert delete_enrollment("nobody", base_dir=tmp_path) is False


def test_enroll_from_samples_rejects_empty_sample(tmp_path):
    try:
        enroll_from_samples("eve", [[0.1], [], [0.3]], base_dir=tmp_path)
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty enrollment sample")


def test_enrollment_json_is_written_for_ui_consumption(tmp_path):
    enroll_from_samples("frank", [_sample(13), _sample(14), _sample(15)], base_dir=tmp_path)

    payload = json.loads((tmp_path / "frank" / "voice_embeddings.json").read_text(encoding="utf-8"))

    assert payload["sample_count"] == 3
    assert payload["model_id"] == "synthetic"
    assert isinstance(payload["embedding"], list)
