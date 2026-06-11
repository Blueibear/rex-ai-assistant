"""Shared type aliases and sentinels for the voice pipeline — extracted verbatim from ``rex/voice_loop.py`` (US-REM-028)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeAlias

from rex.voice.optional_imports import (
    np,
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


_USE_CONFIG_LANGUAGE = object()

if TYPE_CHECKING:
    from numpy.typing import NDArray

    AudioArray: TypeAlias = NDArray[Any]
else:
    AudioArray: TypeAlias = Any

RecorderCallable = Callable[[float], Awaitable[AudioArray] | AudioArray]
IdentifySpeakerCallable = Callable[[AudioArray], str | None] | Callable[[], str | None]

# Backwards-compatible runtime alias used by optional-import tests.
_NDArray = np.ndarray if np is not None else Any
