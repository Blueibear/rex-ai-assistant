from __future__ import annotations

import sys
import types

import pytest

from rex.wakeword import assets as wake_assets
from rex.wakeword import utils as wake_utils


class FailingWakeModel:
    def __init__(self, *args, **kwargs):  # noqa: D401, ANN002, ANN003 - third-party-like signature
        raise RuntimeError("bad model")


@pytest.mark.unit
def test_invalid_custom_onnx_falls_back_when_enabled(tmp_path, monkeypatch):
    model_path = tmp_path / "wake_words" / "hey_rex" / "model.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"broken")

    monkeypatch.setattr(
        wake_utils,
        "resolve_custom_wakeword_model_path",
        lambda phrase, model_path_arg=None: model_path,
    )
    monkeypatch.setattr(wake_utils, "_get_openwakeword", lambda: (object(), FailingWakeModel))
    monkeypatch.setattr(
        wake_utils,
        "_load_builtin_openwakeword_model",
        lambda **kwargs: ("builtin-model", "hey jarvis"),
    )

    model, selection = wake_utils.load_wakeword_model_with_metadata(
        keyword="hey rex",
        backend="custom_onnx",
        fallback_to_builtin=True,
        fallback_keyword="hey jarvis",
    )

    assert model == "builtin-model"
    assert selection.active_backend == "openwakeword"
    assert selection.used_fallback is True
    assert selection.validation_error == "bad model"


@pytest.mark.unit
def test_install_custom_embedding_asset_copies_and_writes_phrase(tmp_path, monkeypatch):
    source = tmp_path / "source.pt"
    source.write_bytes(b"embedding")
    target = tmp_path / "wake_words" / "hey_rex" / "embedding.pt"

    monkeypatch.setattr(
        wake_assets,
        "resolve_custom_wakeword_embedding_path",
        lambda phrase, target_path=None: target,
    )
    monkeypatch.setattr(
        wake_assets,
        "load_wakeword_model_with_metadata",
        lambda **kwargs: (
            object(),
            wake_utils.WakeWordModelSelection(
                requested_backend="custom_embedding",
                active_backend="custom_embedding",
                requested_phrase="hey rex",
                active_label="embedding",
                resolved_embedding_path=str(target),
            ),
        ),
    )

    result = wake_assets.install_custom_wakeword_asset(
        backend="custom_embedding",
        phrase="hey rex",
        source_path=str(source),
    )

    assert result["ok"] is True
    assert target.read_bytes() == b"embedding"
    assert (target.parent / "phrase.txt").read_text(encoding="utf-8") == "hey rex"


@pytest.mark.unit
def test_create_openwakeword_model_asset_writes_default_target(tmp_path, monkeypatch):
    target = tmp_path / "wake_words" / "hey_rex" / "model.onnx"

    class DummyRecorder:
        def __init__(self, backend):  # noqa: D401, ANN001 - third-party-like signature
            assert backend == "onnx"

        def record_wakeword(self, phrase, destination):
            assert phrase == "hey rex"
            with open(destination, "wb") as fh:
                fh.write(b"onnx")

    dummy_openwakeword = types.SimpleNamespace(Model=DummyRecorder)
    monkeypatch.setitem(sys.modules, "openwakeword", dummy_openwakeword)
    monkeypatch.setattr(
        wake_assets,
        "resolve_custom_wakeword_model_path",
        lambda phrase, target_path=None: target,
    )
    monkeypatch.setattr(
        wake_assets,
        "load_wakeword_model_with_metadata",
        lambda **kwargs: (
            object(),
            wake_utils.WakeWordModelSelection(
                requested_backend="custom_onnx",
                active_backend="custom_onnx",
                requested_phrase="hey rex",
                active_label="model",
                resolved_model_path=str(target),
            ),
        ),
    )

    result = wake_assets.create_openwakeword_model_asset(phrase="hey rex")

    assert result["ok"] is True
    assert target.read_bytes() == b"onnx"
    assert (target.parent / "phrase.txt").read_text(encoding="utf-8") == "hey rex"
