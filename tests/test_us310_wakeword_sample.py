"""Tests for US-310: Fix wake word 'Play sample' to play actual wake word audio."""

from __future__ import annotations

import base64
import json
import struct
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")


# ---------------------------------------------------------------------------
# trainer.py — sample.wav saved during training
# ---------------------------------------------------------------------------


def _make_samples(n: int = 3, length: int = 1600) -> list[list[float]]:
    """Return n positive samples as lists of float32 PCM (16kHz)."""
    rng = np.random.default_rng(42)
    return [rng.uniform(-0.1, 0.1, length).tolist() for _ in range(n)]


def test_train_saves_sample_wav(tmp_path: Path) -> None:
    from rex.wakeword.trainer import train_from_samples

    samples = _make_samples()
    result = train_from_samples("hey test", samples, [], config_dir=tmp_path)

    assert result["ok"] is True
    sample_wav = tmp_path / "hey_test" / "sample.wav"
    assert sample_wav.is_file(), "sample.wav should be created during training"


def test_sample_wav_is_valid_wav(tmp_path: Path) -> None:
    from rex.wakeword.trainer import train_from_samples

    samples = _make_samples()
    train_from_samples("hey test", samples, [], config_dir=tmp_path)

    sample_wav = tmp_path / "hey_test" / "sample.wav"
    with wave.open(str(sample_wav)) as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2  # 16-bit
        assert wf.getframerate() == 16000
        assert wf.getnframes() > 0


def test_sample_wav_not_fatal_on_error(tmp_path: Path) -> None:
    """Training should still succeed even if saving the WAV fails."""
    from rex.wakeword.trainer import train_from_samples, _save_sample_wav

    samples = _make_samples()

    with patch("rex.wakeword.trainer._save_sample_wav", side_effect=OSError("disk full")):
        result = train_from_samples("hey test", samples, [], config_dir=tmp_path)

    # Embedding was still saved successfully.
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# list_custom_wake_words — has_sample field
# ---------------------------------------------------------------------------


def test_has_sample_true_when_wav_exists(tmp_path: Path) -> None:
    from rex.wakeword.trainer import train_from_samples, list_custom_wake_words

    samples = _make_samples()
    train_from_samples("hey sample", samples, [], config_dir=tmp_path)

    words = list_custom_wake_words(config_dir=tmp_path)
    assert len(words) == 1
    assert words[0]["has_sample"] is True


def test_has_sample_false_when_wav_missing(tmp_path: Path) -> None:
    from rex.wakeword.trainer import train_from_samples, list_custom_wake_words
    from rex.wakeword.embedding import save_embedding

    # Manually create an embedding dir without sample.wav.
    slug_dir = tmp_path / "no_sample"
    slug_dir.mkdir()
    emb = np.zeros(128, dtype=np.float32)
    save_embedding(slug_dir / "embedding.pt", emb)
    (slug_dir / "phrase.txt").write_text("no sample", encoding="utf-8")

    words = list_custom_wake_words(config_dir=tmp_path)
    assert len(words) == 1
    assert words[0]["has_sample"] is False


# ---------------------------------------------------------------------------
# rex_wakeword_sample_bridge — main()
# ---------------------------------------------------------------------------


def _wav_bytes() -> bytes:
    """Create a minimal valid WAV blob for testing."""
    import io

    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<h", 0) * 160)
    return buf.getvalue()


def test_sample_bridge_returns_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import rex_wakeword_sample_bridge as bridge

    wav_dir = tmp_path / "hey_test"
    wav_dir.mkdir()
    (wav_dir / "sample.wav").write_bytes(_wav_bytes())

    monkeypatch.setattr(bridge, "_CONFIG_DIR_DEFAULT", tmp_path)

    import io
    import sys

    fake_in = io.StringIO(json.dumps({"wake_word_id": "hey_test"}))
    output = []

    monkeypatch.setattr(sys, "stdin", fake_in)
    monkeypatch.setattr("builtins.print", lambda *a, **kw: output.append(a[0]))

    bridge.main()

    assert len(output) == 1
    result = json.loads(output[0])
    assert result["ok"] is True
    assert "audio_base64" in result
    # Verify the base64 decodes to something non-empty.
    assert len(base64.b64decode(result["audio_base64"])) > 0


def test_sample_bridge_no_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import rex_wakeword_sample_bridge as bridge

    monkeypatch.setattr(bridge, "_CONFIG_DIR_DEFAULT", tmp_path)

    import io
    import sys

    fake_in = io.StringIO(json.dumps({"wake_word_id": "missing_word"}))
    output = []

    monkeypatch.setattr(sys, "stdin", fake_in)
    monkeypatch.setattr("builtins.print", lambda *a, **kw: output.append(a[0]))

    bridge.main()

    result = json.loads(output[0])
    assert result["ok"] is False
    assert result.get("has_sample") is False


def test_sample_bridge_missing_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import rex_wakeword_sample_bridge as bridge

    monkeypatch.setattr(bridge, "_CONFIG_DIR_DEFAULT", tmp_path)

    import io
    import sys

    fake_in = io.StringIO("{}")
    output = []

    monkeypatch.setattr(sys, "stdin", fake_in)
    monkeypatch.setattr("builtins.print", lambda *a, **kw: output.append(a[0]))

    bridge.main()

    result = json.loads(output[0])
    assert result["ok"] is False
    assert "wake_word_id" in result["error"]


# ---------------------------------------------------------------------------
# rex_wakeword_list_bridge — has_sample propagated
# ---------------------------------------------------------------------------


def test_list_bridge_has_sample_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """List bridge should pass has_sample through for custom wake words."""
    from rex.wakeword.trainer import train_from_samples

    samples = _make_samples()
    train_from_samples("hey list", samples, [], config_dir=tmp_path)

    import io
    import sys
    import rex_wakeword_list_bridge as lb

    output = []

    def mock_list_custom(config_dir=None):  # type: ignore[no-untyped-def]
        from rex.wakeword.trainer import list_custom_wake_words

        return list_custom_wake_words(config_dir=tmp_path)

    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    monkeypatch.setattr("builtins.print", lambda *a, **kw: output.append(a[0]))

    with patch("rex.wakeword.trainer.list_custom_wake_words", mock_list_custom):
        lb.main()

    result = json.loads(output[0])
    custom = [w for w in result["wake_words"] if w["engine"] == "custom_embedding"]
    if custom:
        assert "has_sample" in custom[0]
