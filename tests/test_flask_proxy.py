"""Tests covering the Flask proxy service."""

from __future__ import annotations

import importlib
import sys

import pytest

pytest.importorskip("flask")


def _load_app(monkeypatch):
    monkeypatch.setenv("REX_PROXY_ALLOW_LOCAL", "1")
    monkeypatch.setenv("USER_ID", "james")

    import rex
    import rex.config

    rex.config.get_settings.cache_clear()
    new_settings = rex.config.get_settings()
    monkeypatch.setattr("rex.config.settings", new_settings, raising=False)
    monkeypatch.setattr("rex.settings", new_settings, raising=False)

    if "flask_proxy" in sys.modules:
        module = importlib.reload(sys.modules["flask_proxy"])
    else:
        module = importlib.import_module("flask_proxy")
    module.PLUGINS["web_search"] = type("P", (), {"name": "web_search", "process": lambda self, ctx: "result", "shutdown": lambda self: None, "initialise": lambda self: None})()
    return module.app, module


def test_whoami_returns_sanitised_profile(monkeypatch):
    app, module = _load_app(monkeypatch)

    with app.test_client() as client:
        response = client.get("/whoami", environ_overrides={"REMOTE_ADDR": "127.0.0.1"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload.get("user") == module.user_key
    assert "profile" in payload
    assert payload["profile"].get("name")


def test_search_validates_query(monkeypatch):
    app, module = _load_app(monkeypatch)

    with app.test_client() as client:
        response = client.get("/search", environ_overrides={"REMOTE_ADDR": "127.0.0.1"})
    assert response.status_code == 400

    with app.test_client() as client:
        response = client.get(
            "/search",
            query_string={"q": "hello"},
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )
    assert response.status_code == 200
    assert response.get_json()["result"] == "result"
