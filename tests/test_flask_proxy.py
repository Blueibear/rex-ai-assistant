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
    assert payload.get("user") == module.g.user_key, "Incorrect user in /whoami response"
    assert "profile" in payload, "Missing profile in /whoami"
    assert payload["profile"].get("name"), "Profile name is missing in /whoami"


def test_whoami_rejects_nonlocal(monkeypatch):
    """Test that /whoami blocks non-local requests if proxy is restricted."""
    monkeypatch.setenv("REX_PROXY_ALLOW_LOCAL", "1")
    app, _ = _load_app(monkeypatch)

    with app.test_client() as client:
        response = client.get("/whoami", environ_overrides={"REMOTE_ADDR": "8.8.8.8"})

    assert response.status_code == 403, "Should reject non-local access with 403"


def test_search_returns_error_when_plugin_missing(monkeypatch):
    """Test fallback behavior when web_search plugin is not installed."""
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package=None):
        if name == "plugins.web_search":
            return None
        return original_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    sys.modules.pop("flask_proxy", None)  # Force reload

    app, _ = _load_app(monkeypatch)

    with app.test_client() as client:
        response = client.get(
            "/search",
            query_string={"q": "open source"},
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )

    assert response.status_code == 503
    payload = response.get_json()
    assert payload["error"]["message"].startswith("Web search plugin is not installed")


def test_search_uses_plugin_when_available(monkeypatch):
    """Test that /search uses the plugin correctly when available."""
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


# --- Dashboard / proxy auth alignment regression tests ---


def test_voice_mode_not_blocked_by_proxy_auth(monkeypatch):
    """Regression: /api/voice/mode must not be blocked by the outer proxy auth layer.

    Before the fix, _DASHBOARD_PREFIXES did not include /api/voice, so the
    load_user_memory() hook would intercept the request and return 403 before
    the dashboard's own @require_auth could handle it.
    """
    app, module = _load_app(monkeypatch)

    # Sanity-check that /api/voice is in the exemption list
    assert any(
        p == "/api/voice" for p in module._DASHBOARD_PREFIXES
    ), "/api/voice must be in _DASHBOARD_PREFIXES so the dashboard's own auth handles it"

    with app.test_client() as client:
        # Request without any proxy credentials — the proxy layer must pass this
        # through rather than returning 403.  The dashboard's own auth will then
        # decide the response (401 or 200 depending on session).
        response = client.get(
            "/api/voice/mode",
            environ_overrides={"REMOTE_ADDR": "8.8.8.8"},
        )

    # The outer proxy must not return 403; dashboard auth returns 401 for no token.
    assert response.status_code != 403, (
        "/api/voice/mode was blocked by the outer proxy layer (403). "
        "It should be delegated to the dashboard's own auth."
    )


def test_dashboard_prefixes_cover_all_dashboard_api_routes(monkeypatch):
    """Verify the set of dashboard-owned API prefixes is consistent with the routes."""
    app, module = _load_app(monkeypatch)

    expected_prefixes = {
        "/dashboard",
        "/api/dashboard",
        "/api/settings",
        "/api/chat",
        "/api/scheduler",
        "/api/notifications",
        "/api/voice",
    }

    actual = set(module._DASHBOARD_PREFIXES)
    missing = expected_prefixes - actual
    assert not missing, f"These prefixes are missing from _DASHBOARD_PREFIXES: {missing}"
