"""Regression tests for rex.voice_loop optional-dependency import safety.

Ensures that rex.voice_loop can be imported and collected by pytest even
when optional heavy dependencies (numpy, simpleaudio, sounddevice) are not
installed.  The root cause being guarded: module-level type alias assignments
referenced np.ndarray unconditionally, causing an AttributeError when np was
None (numpy absent).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

import pytest


def _make_find_spec_without(blocked: str):
    """Return a patched find_spec that reports *blocked* as unresolvable."""
    original = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == blocked:
            return None
        return original(name, *args, **kwargs)

    return fake_find_spec


@pytest.mark.unit
def test_voice_loop_imports_without_numpy(monkeypatch):
    """rex.voice_loop must be importable when numpy is not installed.

    This is the regression guard for the collection error reported in
    Cycle 5.2b verification: module-level type aliases referenced np.ndarray
    unconditionally, so importing the module raised AttributeError when numpy
    was absent.
    """
    # Remove cached modules so the import path is exercised fresh.
    monkeypatch.delitem(sys.modules, "numpy", raising=False)
    monkeypatch.delitem(sys.modules, "rex.voice_loop", raising=False)

    # Block numpy via find_spec so _lazy_import_numpy() returns None.
    monkeypatch.setattr(importlib.util, "find_spec", _make_find_spec_without("numpy"))

    # This must not raise AttributeError or any other ImportError.
    import rex.voice_loop as vl

    assert vl.VoiceLoop is not None
    assert vl.RecorderCallable is not None
    assert vl.IdentifySpeakerCallable is not None


@pytest.mark.unit
def test_voice_loop_type_aliases_use_any_when_numpy_absent(monkeypatch):
    """When numpy is missing, _NDArray falls back to typing.Any."""
    from typing import Any

    monkeypatch.delitem(sys.modules, "numpy", raising=False)
    monkeypatch.delitem(sys.modules, "rex.voice_loop", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", _make_find_spec_without("numpy"))

    import rex.voice_loop as vl

    assert vl._NDArray is Any


@pytest.mark.unit
def test_voice_loop_type_aliases_use_ndarray_when_numpy_present(monkeypatch):
    """When numpy is installed, _NDArray is np.ndarray (not Any)."""
    numpy = pytest.importorskip("numpy")

    # Force a fresh import with numpy available so the module-level assignment
    # runs again (prior tests may leave a cached module with numpy blocked).
    monkeypatch.delitem(sys.modules, "rex.voice_loop", raising=False)

    import rex.voice_loop as vl

    assert vl._NDArray is numpy.ndarray
