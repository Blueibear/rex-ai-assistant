"""Tests covering the Flask proxy service."""

from __future__ import annotations

import importlib
import sys

import pytest

pytest.importorskip("flask")


def _load_app(monkeypatch):
    monkeypatch.setenv("REX_TESTING", "true")
    monkeypatch.setenv("REX_PROXY_ALLOW_LOCAL", "1")
    monkeypatch.setenv("REX_ACTIVE_USER", "james")

    if "flask_proxy" in sys.modules:
        module = importlib.reload(sys.modules["flask_proxy"])
    else:
        module = importlib.import_module("flask_proxy")

    return module.app, module


def test_whoami_returns_sanitised_profile(monkeypatch):
    """Test that /whoami returns user info without memory details."""
    app, module = _load_app(monkeypatch)

    with app.test_client() as client:
        response = client.get("/whoami", environ_overrides={"REMOTE_ADDR": "127.0.0.1"})

    assert response.status_code == 200, "Expected 200 OK from /whoami"
    payload = response.get_json()

    assert "memory" not in payload, "Should not expose memory in /whoami"
    assert payload.get("user") == module.user_key, "Incorrect user in /whoami response"
    assert "profile" in payload, "Missing profile in /whoami"
    assert payload["profile"].get("name"), "Profile name is missing in /whoami"


def test_whoami_rejects_nonlocal(monkeypatch):
    """Test that /whoami blocks non-local requests if proxy is restricted."""
    monkeypatch.setenv("REX_PROXY_ALLOW_LOCAL", "1")  # Enforce local-only
    app, _ = _load_app(monkeypatch)

    with app.test_client() as client:
        response = client.get("/whoami", environ_overrides={"REMOTE_ADDR": "8.8.8.8"})  # Simulate external

    assert response.status_code == 403, "Should reject non-local access with 403"

