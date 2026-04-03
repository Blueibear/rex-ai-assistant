"""Tests for US-WW-002: Custom wake word training."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_samples(n: int, length: int = 1024) -> list[list[float]]:
    """Generate n random audio sample frames as list[list[float]]."""
    rng = np.random.default_rng(42)
    return [rng.standard_normal(length).tolist() for _ in range(n)]


# ---------------------------------------------------------------------------
# rex.wakeword.trainer — unit tests
# ---------------------------------------------------------------------------


def test_train_from_samples_creates_embedding(tmp_path: Path) -> None:
    from rex.wakeword.trainer import train_from_samples

    result = train_from_samples(
        "hey rex",
        _make_samples(5),
        _make_samples(3),
        config_dir=tmp_path,
    )
    assert result["ok"] is True
    embedding_path = Path(result["model_path"])
    assert embedding_path.is_file()
    assert embedding_path.stat().st_size > 0


def test_train_from_samples_saves_phrase_meta(tmp_path: Path) -> None:
    from rex.wakeword.trainer import train_from_samples

    train_from_samples("hey rex", _make_samples(5), [], config_dir=tmp_path)
    meta = (tmp_path / "hey_rex" / "phrase.txt").read_text(encoding="utf-8")
    assert meta == "hey rex"


def test_train_from_samples_slug_with_spaces(tmp_path: Path) -> None:
    from rex.wakeword.trainer import train_from_samples

    result = train_from_samples("ok computer", _make_samples(5), [], config_dir=tmp_path)
    assert result["ok"] is True
    assert "ok_computer" in result["model_path"]


def test_train_from_samples_requires_min_positives(tmp_path: Path) -> None:
    from rex.wakeword.trainer import train_from_samples

    result = train_from_samples("hey rex", _make_samples(2), [], config_dir=tmp_path)
    assert result["ok"] is False
    assert "positive" in result["error"].lower()


def test_train_from_samples_empty_phrase(tmp_path: Path) -> None:
    from rex.wakeword.trainer import train_from_samples

    result = train_from_samples("", _make_samples(5), [], config_dir=tmp_path)
    assert result["ok"] is False


def test_train_from_samples_whitespace_phrase(tmp_path: Path) -> None:
    from rex.wakeword.trainer import train_from_samples

    result = train_from_samples("   ", _make_samples(5), [], config_dir=tmp_path)
    assert result["ok"] is False


def test_train_produces_loadable_embedding(tmp_path: Path) -> None:
    from rex.wakeword.embedding import load_embedding
    from rex.wakeword.trainer import train_from_samples

    result = train_from_samples("hey rex", _make_samples(5), [], config_dir=tmp_path)
    emb = load_embedding(result["model_path"])
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1


# ---------------------------------------------------------------------------
# list_custom_wake_words
# ---------------------------------------------------------------------------


def test_list_custom_wake_words_empty_dir(tmp_path: Path) -> None:
    from rex.wakeword.trainer import list_custom_wake_words

    result = list_custom_wake_words(config_dir=tmp_path)
    assert result == []


def test_list_custom_wake_words_missing_dir(tmp_path: Path) -> None:
    from rex.wakeword.trainer import list_custom_wake_words

    result = list_custom_wake_words(config_dir=tmp_path / "nonexistent")
    assert result == []


def test_list_custom_wake_words_returns_trained(tmp_path: Path) -> None:
    from rex.wakeword.trainer import list_custom_wake_words, train_from_samples

    train_from_samples("hey rex", _make_samples(5), [], config_dir=tmp_path)
    entries = list_custom_wake_words(config_dir=tmp_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["id"] == "hey_rex"
    assert entry["engine"] == "custom_embedding"
    assert "embedding.pt" in entry["model_path"]


def test_list_custom_wake_words_uses_phrase_txt(tmp_path: Path) -> None:
    from rex.wakeword.trainer import list_custom_wake_words, train_from_samples

    train_from_samples("hey rex", _make_samples(5), [], config_dir=tmp_path)
    entries = list_custom_wake_words(config_dir=tmp_path)
    assert entries[0]["name"] == "hey rex"


def test_list_custom_wake_words_skips_dirs_without_embedding(tmp_path: Path) -> None:
    from rex.wakeword.trainer import list_custom_wake_words

    (tmp_path / "stale_entry").mkdir()
    entries = list_custom_wake_words(config_dir=tmp_path)
    assert entries == []


# ---------------------------------------------------------------------------
# rex_wakeword_train_bridge — black-box via in-process invocation
# ---------------------------------------------------------------------------


def _run_train_bridge(payload: dict) -> dict:
    import contextlib
    import io

    captured = io.StringIO()
    original_stdin = sys.stdin
    sys.stdin = io.StringIO(json.dumps(payload))
    try:
        with contextlib.redirect_stdout(captured):
            import rex_wakeword_train_bridge  # noqa: PLC0415

            try:
                rex_wakeword_train_bridge.main()
            except SystemExit:
                pass
    finally:
        sys.stdin = original_stdin
    return json.loads(captured.getvalue().strip())


def test_bridge_trains_successfully(tmp_path: Path) -> None:
    samples = _make_samples(5)
    with patch("rex.wakeword.trainer._CONFIG_DIR_DEFAULT", tmp_path):
        result = _run_train_bridge(
            {"phrase": "hey test", "positive_samples": samples, "negative_samples": []}
        )
    assert result["ok"] is True
    assert result["phrase"] == "hey test"
    assert "model_path" in result


def test_bridge_returns_error_on_empty_phrase() -> None:
    result = _run_train_bridge(
        {"phrase": "", "positive_samples": _make_samples(5), "negative_samples": []}
    )
    assert result["ok"] is False
    assert "error" in result


def test_bridge_returns_error_on_too_few_samples() -> None:
    result = _run_train_bridge(
        {"phrase": "hey rex", "positive_samples": _make_samples(1), "negative_samples": []}
    )
    assert result["ok"] is False
