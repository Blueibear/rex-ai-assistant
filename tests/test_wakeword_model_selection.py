from __future__ import annotations

import logging
import sys
import types

import pytest

from rex.config import AppConfig, build_app_config
from rex.wakeword import listener as wake_listener
from rex.wakeword import utils as wake_utils


class DummyWakeModel:
    def __init__(self, wakeword_models, inference_framework, enable_speex_noise_suppression=False):
        self.wakeword_models = wakeword_models
        self.inference_framework = inference_framework
        self.enable_speex_noise_suppression = enable_speex_noise_suppression

    def predict(self, audio_frame):
        return {"dummy": 0.0}


def _install_dummy_openwakeword(monkeypatch, models):
    dummy_openwakeword = types.SimpleNamespace(MODELS=models)
    monkeypatch.setattr(wake_utils, "openwakeword", dummy_openwakeword)
    monkeypatch.setattr(wake_utils, "WakeWordModel", DummyWakeModel)

    openwakeword_module = types.ModuleType("openwakeword")
    openwakeword_module.MODELS = models

    utils_module = types.ModuleType("openwakeword.utils")

    def download_models(model_names):  # noqa: ARG001 - signature matches openwakeword
        return None

    utils_module.download_models = download_models

    monkeypatch.setitem(sys.modules, "openwakeword", openwakeword_module)
    monkeypatch.setitem(sys.modules, "openwakeword.utils", utils_module)


@pytest.mark.unit
def test_get_openwakeword_resolves_model_class_from_restored_module(monkeypatch):
    restored = types.SimpleNamespace(
        MODELS={"hey_jarvis": object()},
        model=types.SimpleNamespace(Model=DummyWakeModel),
    )
    monkeypatch.setattr(wake_utils, "openwakeword", restored)
    monkeypatch.setattr(wake_utils, "WakeWordModel", object)
    monkeypatch.setattr(wake_utils, "_OPENWAKEWORD_MODULE", object())
    monkeypatch.setattr(wake_utils, "_WAKEWORD_MODEL", object)

    module, model_cls = wake_utils._get_openwakeword()

    assert module is restored
    assert model_cls is DummyWakeModel


@pytest.mark.unit
def test_get_openwakeword_refreshes_stale_cache_when_alias_removed(monkeypatch):
    stale = types.SimpleNamespace(MODELS={})
    fresh = types.SimpleNamespace(
        MODELS={"hey_jarvis": object()},
        model=types.SimpleNamespace(Model=DummyWakeModel),
    )
    monkeypatch.setattr(wake_utils, "openwakeword", None)
    monkeypatch.setattr(wake_utils, "_OPENWAKEWORD_MODULE", stale)
    monkeypatch.setattr(wake_utils, "_WAKEWORD_MODEL", object)
    monkeypatch.delitem(sys.modules, "openwakeword", raising=False)
    monkeypatch.setattr(wake_utils, "_lazy_import_openwakeword", lambda: (fresh, DummyWakeModel))

    module, model_cls = wake_utils._get_openwakeword()

    assert module is fresh
    assert model_cls is DummyWakeModel


@pytest.mark.unit
def test_openwakeword_invalid_keyword_falls_back_to_hey_jarvis(monkeypatch, caplog):
    models = {
        "hey_jarvis": {
            "model_path": "hey_jarvis.tflite",
            "download_url": "http://example.com/hey_jarvis.zip",
        },
    }
    _install_dummy_openwakeword(monkeypatch, models)

    caplog.set_level(logging.WARNING)
    _, label = wake_utils.load_wakeword_model(keyword="rex", backend="openwakeword")

    assert label == "hey jarvis"
    assert any("Falling back" in record.message for record in caplog.records)


@pytest.mark.unit
def test_openwakeword_invalid_keyword_falls_back_to_first_available(monkeypatch):
    models = {
        "alexa": {"model_path": "alexa.tflite", "download_url": "http://example.com/alexa.zip"},
        "timer": {"model_path": "timer.tflite", "download_url": "http://example.com/timer.zip"},
    }
    _install_dummy_openwakeword(monkeypatch, models)

    _, label = wake_utils.load_wakeword_model(keyword="rex", backend="openwakeword")

    assert label == "alexa"


@pytest.mark.unit
def test_openwakeword_no_available_keywords_raises(monkeypatch):
    _install_dummy_openwakeword(monkeypatch, {})

    with pytest.raises(RuntimeError, match="No openwakeword models available"):
        wake_utils.load_wakeword_model(keyword="rex", backend="openwakeword")


@pytest.mark.unit
def test_wakeword_keyword_alias():
    config = build_app_config({"wake_word": {"wakeword": "rex", "keyword": None}})
    assert config.wakeword_keyword == "rex"


@pytest.mark.unit
def test_custom_onnx_missing_model_falls_back(monkeypatch):
    models = {
        "hey_jarvis": {
            "model_path": "hey_jarvis.tflite",
            "download_url": "http://example.com/hey_jarvis.zip",
        },
    }
    _install_dummy_openwakeword(monkeypatch, models)

    _, label = wake_utils.load_wakeword_model(
        backend="custom_onnx",
        model_path="missing.onnx",
        fallback_to_builtin=True,
        fallback_keyword="hey jarvis",
    )

    assert label == "hey jarvis"


@pytest.mark.unit
def test_custom_onnx_missing_model_raises(monkeypatch):
    _install_dummy_openwakeword(monkeypatch, {})

    with pytest.raises(RuntimeError, match="Custom wake word model not found"):
        wake_utils.load_wakeword_model(
            backend="custom_onnx",
            model_path="missing.onnx",
            fallback_to_builtin=False,
        )


@pytest.mark.unit
def test_custom_embedding_missing_model_falls_back(monkeypatch):
    models = {
        "hey_jarvis": {
            "model_path": "hey_jarvis.tflite",
            "download_url": "http://example.com/hey_jarvis.zip",
        },
    }
    _install_dummy_openwakeword(monkeypatch, models)

    _, label = wake_utils.load_wakeword_model(
        backend="custom_embedding",
        embedding_path="missing.pt",
        fallback_to_builtin=True,
        fallback_keyword="hey jarvis",
    )

    assert label == "hey jarvis"


@pytest.mark.unit
def test_custom_embedding_missing_model_raises(monkeypatch):
    _install_dummy_openwakeword(monkeypatch, {})

    with pytest.raises(RuntimeError, match="Custom wake word embedding not found"):
        wake_utils.load_wakeword_model(
            backend="custom_embedding",
            embedding_path="missing.pt",
            fallback_to_builtin=False,
        )


@pytest.mark.unit
def test_async_assistant_init_with_invalid_keyword(monkeypatch):
    models = {
        "hey_jarvis": {
            "model_path": "hey_jarvis.tflite",
            "download_url": "http://example.com/hey_jarvis.zip",
        },
    }
    _install_dummy_openwakeword(monkeypatch, models)

    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    monkeypatch.setitem(sys.modules, "sounddevice", types.ModuleType("sounddevice"))
    monkeypatch.setitem(sys.modules, "soundfile", types.ModuleType("soundfile"))

    whisper_module = types.ModuleType("whisper")
    whisper_module.load_model = lambda model_name: None
    monkeypatch.setitem(sys.modules, "whisper", whisper_module)

    tts_module = types.ModuleType("TTS")
    tts_api_module = types.ModuleType("TTS.api")

    class DummyTTS:
        def __init__(self, *args, **kwargs):  # noqa: D401, ANN001 - matches third party API
            pass

    tts_api_module.TTS = DummyTTS
    monkeypatch.setitem(sys.modules, "TTS", tts_module)
    monkeypatch.setitem(sys.modules, "TTS.api", tts_api_module)

    import voice_loop as voice_loop_module

    class DummyLanguageModel:
        def __init__(self, config):  # noqa: ARG002 - matches LanguageModel signature
            self.config = config

    monkeypatch.setattr(voice_loop_module, "LanguageModel", DummyLanguageModel)
    monkeypatch.setattr(
        voice_loop_module, "load_audio_config", lambda: {"input_device_index": None}
    )
    monkeypatch.setattr(
        voice_loop_module,
        "resolve_audio_device",
        lambda configured_device, sample_rate: (None, "ok"),
    )
    monkeypatch.setattr(voice_loop_module, "_load_plugins_impl", lambda: {})
    monkeypatch.setattr(voice_loop_module, "load_users_map", lambda: {})
    monkeypatch.setattr(voice_loop_module, "load_all_profiles", lambda: {})
    monkeypatch.setattr(
        voice_loop_module, "resolve_user_key", lambda user, users_map, profiles=None: user
    )
    monkeypatch.setattr(
        voice_loop_module, "extract_voice_reference", lambda profile, user_key=None: None
    )

    config = AppConfig(wakeword="rex", wakeword_keyword=None, wakeword_backend="openwakeword")

    assistant = voice_loop_module.AsyncRexAssistant(config)
    assert assistant._wake_keyword == "hey jarvis"


@pytest.mark.unit
def test_resolve_custom_wakeword_model_path_uses_slugged_phrase():
    resolved = wake_utils.resolve_custom_wakeword_model_path("Hey Rex!")

    assert resolved.name == "model.onnx"
    assert resolved.parent.name == "hey_rex"


@pytest.mark.unit
def test_load_custom_onnx_selection_uses_default_path_when_model_exists(tmp_path, monkeypatch):
    model_path = tmp_path / "wake_words" / "hey_rex" / "model.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"fake-onnx")

    monkeypatch.setattr(
        wake_utils,
        "resolve_custom_wakeword_model_path",
        lambda phrase, model_path_arg=None: model_path,
    )
    monkeypatch.setattr(wake_utils, "_get_openwakeword", lambda: (object(), DummyWakeModel))

    model, selection = wake_utils.load_wakeword_model_with_metadata(
        keyword="hey rex",
        backend="custom_onnx",
        fallback_to_builtin=True,
        fallback_keyword="hey jarvis",
    )

    assert isinstance(model, DummyWakeModel)
    assert model.wakeword_models == [str(model_path)]
    assert model.inference_framework == "onnx"
    assert selection.requested_backend == "custom_onnx"
    assert selection.active_backend == "custom_onnx"
    assert selection.requested_phrase == "hey rex"
    assert selection.resolved_model_path == str(model_path)
    assert selection.used_fallback is False


@pytest.mark.unit
def test_load_custom_embedding_selection_falls_back_to_builtin_when_asset_missing(
    tmp_path, monkeypatch
):
    missing_embedding = tmp_path / "wake_words" / "hey_rex" / "embedding.pt"

    monkeypatch.setattr(
        wake_utils,
        "resolve_custom_wakeword_embedding_path",
        lambda phrase, embedding_path_arg=None: missing_embedding,
    )
    monkeypatch.setattr(
        wake_utils,
        "_load_builtin_openwakeword_model",
        lambda **kwargs: ("builtin-model", "hey jarvis"),
    )

    model, selection = wake_utils.load_wakeword_model_with_metadata(
        keyword="hey rex",
        backend="custom_embedding",
        fallback_to_builtin=True,
        fallback_keyword="hey jarvis",
    )

    assert model == "builtin-model"
    assert selection.requested_backend == "custom_embedding"
    assert selection.active_backend == "openwakeword"
    assert selection.requested_phrase == "hey rex"
    assert selection.resolved_embedding_path == str(missing_embedding)
    assert selection.used_fallback is True
    assert selection.fallback_keyword == "hey jarvis"


@pytest.mark.unit
def test_build_default_detector_rejects_missing_explicit_custom_onnx(monkeypatch):
    monkeypatch.setattr(
        wake_listener,
        "load_wakeword_model_with_metadata",
        lambda **kwargs: (
            object(),
            wake_utils.WakeWordModelSelection(
                requested_backend="custom_onnx",
                active_backend="openwakeword",
                requested_phrase="hey rex",
                active_label="hey jarvis",
                requested_model_path="missing.onnx",
                resolved_model_path="missing.onnx",
                used_fallback=True,
                fallback_keyword="hey jarvis",
            ),
        ),
    )

    with pytest.raises(wake_listener.WakeWordError, match="not found"):
        wake_listener.build_default_detector(
            sample_rate=16000,
            chunk_duration=1.0,
            threshold=0.2,
            keyword="hey rex",
            model_path="missing.onnx",
            backend="custom_onnx",
            fallback_to_builtin=True,
            fallback_keyword="hey jarvis",
        )
