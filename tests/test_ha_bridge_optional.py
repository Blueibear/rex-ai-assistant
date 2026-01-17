from __future__ import annotations

import importlib
import importlib.util

import pytest

import rex.ha_bridge as ha_bridge


def _force_missing_flask(monkeypatch):
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "flask":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(ha_bridge, "_flask_blueprint", None)
    monkeypatch.setattr(ha_bridge, "_flask_jsonify", None)
    monkeypatch.setattr(ha_bridge, "_flask_request", None)


def test_assistant_imports_without_flask(monkeypatch):
    _force_missing_flask(monkeypatch)
    import rex.assistant as assistant_module

    importlib.reload(assistant_module)
    assert assistant_module.Assistant is not None


def test_create_blueprint_requires_flask(monkeypatch):
    _force_missing_flask(monkeypatch)
    with pytest.raises(
        RuntimeError,
        match="Flask is required for Home Assistant bridge. Install with: pip install flask",
    ):
        ha_bridge.create_blueprint()
