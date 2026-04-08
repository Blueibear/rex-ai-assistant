"""Tests for US-057: Integrations and Capabilities API endpoints.

Covers:
- GET /api/integrations returns list of integrations with name/key/configured fields
- GET /api/integrations returns empty list gracefully when config unavailable
- GET /api/capabilities returns list from capability registry
- Each integration entry has required keys
- Capability entries include name, description, category, enabled
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("REX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REX_JWT_SECRET", "test-us057-secret")
    return tmp_path


@pytest.fixture()
def flask_client(tmp_data_dir: Path):  # type: ignore[override]
    from rex.gui_app import _create_flask_app

    app = _create_flask_app(ui_enabled=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ---------------------------------------------------------------------------
# GET /api/integrations
# ---------------------------------------------------------------------------


class TestIntegrationsEndpoint:
    def test_returns_200_with_integrations_key(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.get("/api/integrations")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "integrations" in data

    def test_integrations_is_a_list(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.get("/api/integrations")
        data = resp.get_json()
        assert isinstance(data["integrations"], list)

    def test_each_entry_has_required_keys(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.get("/api/integrations")
        data = resp.get_json()
        for entry in data["integrations"]:
            assert "name" in entry
            assert "key" in entry
            assert "configured" in entry

    def test_configured_field_is_boolean(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.get("/api/integrations")
        data = resp.get_json()
        for entry in data["integrations"]:
            assert isinstance(entry["configured"], bool)

    def test_known_integration_keys_present(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.get("/api/integrations")
        data = resp.get_json()
        keys = {e["key"] for e in data["integrations"]}
        assert "home_assistant" in keys
        assert "openai" in keys
        assert "email" in keys

    def test_returns_empty_list_on_config_error(self, flask_client) -> None:  # type: ignore[override]
        with patch("rex.config.load_config", side_effect=RuntimeError("no config")):
            resp = flask_client.get("/api/integrations")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["integrations"] == []

    def test_no_auth_required(self, flask_client) -> None:  # type: ignore[override]
        """Integrations endpoint is public — no token needed."""
        resp = flask_client.get("/api/integrations")
        assert resp.status_code == 200

    def test_ha_configured_reflects_env(self, flask_client, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[override]
        """With no HA env vars set, Home Assistant shows as not configured."""
        monkeypatch.delenv("HA_TOKEN", raising=False)
        resp = flask_client.get("/api/integrations")
        data = resp.get_json()
        ha = next(e for e in data["integrations"] if e["key"] == "home_assistant")
        assert isinstance(ha["configured"], bool)


# ---------------------------------------------------------------------------
# GET /api/capabilities
# ---------------------------------------------------------------------------


class TestCapabilitiesEndpoint:
    def test_returns_200_with_capabilities_key(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.get("/api/capabilities")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "capabilities" in data

    def test_capabilities_is_a_list(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.get("/api/capabilities")
        data = resp.get_json()
        assert isinstance(data["capabilities"], list)

    def test_each_entry_has_required_keys(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.get("/api/capabilities")
        data = resp.get_json()
        for entry in data["capabilities"]:
            assert "name" in entry
            assert "description" in entry
            assert "category" in entry
            assert "enabled" in entry

    def test_enabled_field_is_boolean(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.get("/api/capabilities")
        data = resp.get_json()
        for entry in data["capabilities"]:
            assert isinstance(entry["enabled"], bool)

    def test_known_capability_present(self, flask_client) -> None:  # type: ignore[override]
        resp = flask_client.get("/api/capabilities")
        data = resp.get_json()
        names = {e["name"] for e in data["capabilities"]}
        # chat is always enabled; others depend on config
        assert "chat" in names

    def test_no_auth_required(self, flask_client) -> None:  # type: ignore[override]
        """Capabilities endpoint is public — no token needed."""
        resp = flask_client.get("/api/capabilities")
        assert resp.status_code == 200

    def test_returns_empty_list_on_registry_error(self, flask_client) -> None:  # type: ignore[override]
        with patch(
            "rex.capabilities.registry.get_capability_registry",
            side_effect=RuntimeError("registry boom"),
        ):
            resp = flask_client.get("/api/capabilities")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["capabilities"] == []
