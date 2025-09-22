"""Tests covering the Flask proxy service."""

from __future__ import annotations

import importlib
import sys

import pytest

pytest.importorskip("flask")


def _load_app(monkeypatch):
    monkeypatch.setenv("REX_PROXY_ALLOW_LOCAL", "1")
    monkeypatch.setenv("REX_ACTIVE_USER", "james")

    if "flask_proxy" in sys.modules:
        module = importlib.reload(sys.modules["flask_proxy"])
    else:
        module = importlib.import_module("flask_proxy")
    return module.app, module


def test_whoami_returns_sanitised_profile(monkeypatch):
    app, module = _load_app(monkeypatch)

    with app.test_client() as client:
        response = client.get("/whoami", environ_overrides={"REMOTE_ADDR": "127.0.0.1"})

    assert response.status_code == 200
    payload = response.get_json()
    assert "memory" not in payload
    assert payload.get("user") == module.user_key
    assert "profile" in payload
    assert payload["profile"].get("name")


def test_search_returns_error_when_plugin_missing(monkeypatch):
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package=None):  # pragma: no cover - helper
        if name == "plugins.web_search":
            return None
        return original_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    # Force the module to reload with the patched find_spec
    sys.modules.pop("flask_proxy", None)

    app, _module = _load_app(monkeypatch)

    with app.test_client() as client:
        response = client.get(
            "/search",
            query_string={"q": "open source"},
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

    assert response.status_code == 503
    payload = response.get_json()
    assert payload["error"].startswith("Web search plugin is not installed")


def test_search_uses_plugin_when_available(monkeypatch):
    app, module = _load_app(monkeypatch)

    module.search_web = lambda query: {"summary": f"result for {query}"}

    with app.test_client() as client:
        response = client.get(
            "/search",
            query_string={"q": "python"},
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["query"] == "python"
    assert payload["result"] == {"summary": "result for python"}
