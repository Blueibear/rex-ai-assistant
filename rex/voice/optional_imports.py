"""Lazy optional-dependency imports for the voice pipeline — extracted verbatim from ``rex/voice_loop.py`` (US-REM-028)."""

from __future__ import annotations

import importlib.util
import sys
from importlib import import_module

from rex.assistant_errors import (
    AudioDeviceError,
)


def _vl():
    """Return the ``rex.voice_loop`` facade module at call time.

    ``rex.voice_loop`` remains the single patch point for settings, lazy
    importers, audio helpers, and pipeline classes (tests monkeypatch
    ``rex.voice_loop.<name>``). Resolving through the facade at call time
    preserves that behavior without an import cycle at module load time.
    """
    import importlib

    return importlib.import_module("rex.voice_loop")


def _import_optional(module_name: str):
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    # Resolve find_spec at call time: optional-import tests patch
    # ``importlib.util.find_spec`` to simulate missing dependencies.
    if importlib.util.find_spec(module_name) is None:
        return None
    return import_module(module_name)


def _lazy_import_numpy():
    return _import_optional("numpy")


np = _lazy_import_numpy()


def _lazy_import_simpleaudio():
    return _import_optional("simpleaudio")


sa = _lazy_import_simpleaudio()


def _lazy_import_whisper():
    return _import_optional("whisper")


def _lazy_import_tts():
    # Only check availability - do NOT import TTS yet (it triggers
    # internal transformers imports that need the shim first).
    if importlib.util.find_spec("TTS") is None:
        return None
    from rex.compat import ensure_transformers_compatibility

    ensure_transformers_compatibility()
    return import_module("TTS.api").TTS


def _lazy_import_soundfile():
    return _import_optional("soundfile")


def _load_sounddevice():
    # The sounddevice cache lives on the rex.voice_loop facade so that tests
    # can stub it by setting ``rex.voice_loop.sd`` (original module-global
    # behavior, US-REM-028).
    vl = _vl()
    if vl.sd is not None:
        return vl.sd
    vl.sd = _import_optional("sounddevice")
    return vl.sd


def _require_numpy():
    if np is None:
        raise AudioDeviceError("numpy is required for audio processing")
    return np


def _require_sounddevice():
    module = _load_sounddevice()
    if module is None:
        raise AudioDeviceError("sounddevice is not installed")
    return module
