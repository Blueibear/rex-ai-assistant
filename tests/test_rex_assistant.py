"""Targeted unit tests for rex_assistant helpers."""

from __future__ import annotations

import importlib
import importlib
import sys
import types

import pytest


class _DummyStream:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyWaveObject:
    def __init__(self, path):
        self.path = path

    @classmethod
    def from_wave_file(cls, path):
        return cls(path)

    def play(self):
        return types.SimpleNamespace(wait_done=lambda: None)


def _install_stubs(monkeypatch):
    numpy = types.ModuleType("numpy")
    numpy.ndarray = list
    numpy.squeeze = lambda data: data

    openwakeword = types.ModuleType("openwakeword")
    openwakeword.MODELS = {"hey_jarvis": object()}

    class _DummyWakeModel:
        def __init__(self, *args, **kwargs):
            self._threshold = 1.0

        def predict(self, audio_frame):
            return {"hey_jarvis": 1.0}

    openwakeword.model = types.SimpleNamespace(Model=_DummyWakeModel)
    openwakeword_model_module = types.ModuleType("openwakeword.model")
    openwakeword_model_module.Model = _DummyWakeModel

    import importlib.machinery as machinery

    plugins_pkg = types.ModuleType("plugins")
    plugins_pkg.__spec__ = machinery.ModuleSpec("plugins", loader=None)
    plugins_pkg.__path__ = []
    plugins_web_search = types.ModuleType("plugins.web_search")
    plugins_web_search.search_web = lambda query: {"summary": f"stub result for {query}"}
    plugins_web_search.__spec__ = machinery.ModuleSpec("plugins.web_search", loader=None)

    sounddevice = types.ModuleType("sounddevice")
    sounddevice.query_devices = lambda: []
    sounddevice.InputStream = _DummyStream
    sounddevice.rec = (
        lambda frames, samplerate=None, channels=None: [[0.0] for _ in range(frames)]
    )
    sounddevice.wait = lambda: None
    sounddevice.default = types.SimpleNamespace(device=(0, 0))

    simpleaudio = types.ModuleType("simpleaudio")
    simpleaudio.WaveObject = _DummyWaveObject

    soundfile = types.ModuleType("soundfile")
    soundfile.write = lambda filename, data, samplerate: None

    whisper = types.ModuleType("whisper")
    whisper.load_model = lambda name: types.SimpleNamespace(transcribe=lambda path: {"text": "test"})

    tts_api = types.ModuleType("TTS.api")
    tts_api.TTS = object
    tts_pkg = types.ModuleType("TTS")
    tts_pkg.api = tts_api

    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setitem(sys.modules, "openwakeword", openwakeword)
    monkeypatch.setitem(sys.modules, "openwakeword.model", openwakeword_model_module)
    monkeypatch.setitem(sys.modules, "plugins", plugins_pkg)
    monkeypatch.setitem(sys.modules, "plugins.web_search", plugins_web_search)
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
    monkeypatch.setitem(sys.modules, "simpleaudio", simpleaudio)
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    monkeypatch.setitem(sys.modules, "whisper", whisper)
    monkeypatch.setitem(sys.modules, "TTS.api", tts_api)
    monkeypatch.setitem(sys.modules, "TTS", tts_pkg)


@pytest.fixture
def assistant_module(monkeypatch):
    _install_stubs(monkeypatch)
    monkeypatch.setenv("REX_LLM_MODEL", "sshleifer/tiny-gpt2")
    sys.modules.pop("rex_assistant", None)
    module = importlib.import_module("rex_assistant")
    monkeypatch.setattr(module, "LLM", types.SimpleNamespace(generate=lambda **_: "hi"))
    return module


def test_handle_command_routes_function(monkeypatch, assistant_module):
    module = assistant_module
    monkeypatch.setattr(module, "ROUTER", module.FunctionRouter())
    module.ROUTER.register(lambda text: text.startswith("time"), lambda _: "It is noon.")

    result = module.handle_command("time please")

    assert result == "It is noon."


def test_generate_response_fallback(monkeypatch, assistant_module):
    module = assistant_module
    monkeypatch.setattr(module, "LLM", types.SimpleNamespace(generate=lambda **_: (_ for _ in ()).throw(RuntimeError("boom"))))

    response = module.generate_response("Hello there")

    assert "I heard" in response
