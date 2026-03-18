"""US-042: Home Assistant API connection tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rex.ha_bridge import HABridge


def _make_bridge(
    base_url: str = "http://ha.local:8123",
    token: str = "test-token",
    entity_map: dict[str, str] | None = None,
) -> HABridge:
    """Build an HABridge with a mocked requests.Session."""
    with patch("rex.ha_bridge.requests") as mock_requests:
        mock_session = MagicMock()
        mock_requests.Session.return_value = mock_session
        bridge = HABridge(
            base_url=base_url,
            token=token,
            entity_map=entity_map or {},
        )
        bridge._session = mock_session
    return bridge


def _stub_response(bridge: HABridge, json_data: Any, status_code: int = 200) -> MagicMock:
    """Configure bridge._session to return a stub response."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = json_data
    mock_response.content = b"ok"
    mock_response.raise_for_status = MagicMock()
    bridge._session.request.return_value = mock_response
    return mock_response


# ------------------------------------------------------------------ #
# Criterion 1: API reachable
# ------------------------------------------------------------------ #


class TestAPIReachable:
    def test_enabled_when_base_url_and_token_set(self) -> None:
        bridge = _make_bridge()
        assert bridge.enabled is True

    def test_not_enabled_without_base_url(self) -> None:
        bridge = _make_bridge(base_url="", token="tok")
        assert bridge.enabled is False

    def test_not_enabled_without_token(self) -> None:
        bridge = _make_bridge(base_url="http://ha.local:8123", token="")
        assert bridge.enabled is False

    def test_request_reaches_correct_url(self) -> None:
        bridge = _make_bridge(base_url="http://ha.local:8123", token="mytoken")
        _stub_response(bridge, {"message": "API running."})

        bridge._request("GET", "/api/")

        call_kwargs = bridge._session.request.call_args
        assert call_kwargs[1]["url"] == "http://ha.local:8123/api/"

    def test_disabled_bridge_raises_on_request(self) -> None:
        bridge = _make_bridge(base_url="", token="")
        with pytest.raises(RuntimeError, match="not configured"):
            bridge._request("GET", "/api/")


# ------------------------------------------------------------------ #
# Criterion 2: Authentication works
# ------------------------------------------------------------------ #


class TestAuthentication:
    def test_bearer_token_sent_in_header(self) -> None:
        bridge = _make_bridge(token="secret-bearer-token")
        _stub_response(bridge, {})

        bridge._request("GET", "/api/states")

        call_kwargs = bridge._session.request.call_args
        headers = call_kwargs[1]["headers"]
        assert headers["Authorization"] == "Bearer secret-bearer-token"

    def test_content_type_header_set(self) -> None:
        bridge = _make_bridge()
        _stub_response(bridge, {})

        bridge._request("GET", "/api/states")

        call_kwargs = bridge._session.request.call_args
        headers = call_kwargs[1]["headers"]
        assert headers["Content-Type"] == "application/json"

    def test_http_error_propagates(self) -> None:
        bridge = _make_bridge()
        mock_response = MagicMock()
        mock_response.content = b""
        import requests as req_lib

        mock_response.raise_for_status.side_effect = req_lib.HTTPError("401 Unauthorized")
        bridge._session.request.return_value = mock_response

        with pytest.raises(req_lib.HTTPError):
            bridge._request("GET", "/api/states")

    def test_blueprint_secret_validated(self) -> None:
        """Blueprint rejects requests with wrong secret."""
        bridge = _make_bridge()
        bridge._secret = "correct-secret"

        from rex.ha_bridge import create_blueprint

        bp = create_blueprint(bridge)

        import flask

        app = flask.Flask(__name__)
        app.register_blueprint(bp)
        client = app.test_client()

        resp = client.get("/ha/intents", headers={"HASS_SECRET": "wrong-secret"})
        assert resp.status_code == 403

    def test_blueprint_correct_secret_allowed(self) -> None:
        bridge = _make_bridge()
        bridge._secret = "correct-secret"

        from rex.ha_bridge import create_blueprint

        bp = create_blueprint(bridge)

        import flask

        app = flask.Flask(__name__)
        app.register_blueprint(bp)
        client = app.test_client()

        resp = client.get("/ha/intents", headers={"HASS_SECRET": "correct-secret"})
        assert resp.status_code == 200


# ------------------------------------------------------------------ #
# Criterion 3: Entities retrieved
# ------------------------------------------------------------------ #


class TestEntitiesRetrieved:
    def test_list_entities_returns_entities(self) -> None:
        bridge = _make_bridge(
            entity_map={
                "living room light": "light.living_room",
                "kitchen switch": "switch.kitchen",
            }
        )
        _stub_response(
            bridge,
            [
                {
                    "entity_id": "light.living_room",
                    "attributes": {"friendly_name": "Living Room Light"},
                },
                {
                    "entity_id": "switch.kitchen",
                    "attributes": {"friendly_name": "Kitchen Switch"},
                },
            ],
        )

        entities = bridge.list_entities()
        assert len(entities) >= 1

    def test_list_entities_contains_entity_id(self) -> None:
        bridge = _make_bridge(entity_map={"bedroom lamp": "light.bedroom"})
        _stub_response(
            bridge,
            [
                {
                    "entity_id": "light.bedroom",
                    "attributes": {"friendly_name": "Bedroom Lamp"},
                }
            ],
        )

        entities = bridge.list_entities()
        entity_ids = [e["entity_id"] for e in entities]
        assert "light.bedroom" in entity_ids

    def test_list_entities_disabled_raises(self) -> None:
        bridge = _make_bridge(base_url="", token="")
        with pytest.raises(RuntimeError, match="not configured"):
            bridge.list_entities()

    def test_entity_cache_refresh_calls_api(self) -> None:
        bridge = _make_bridge()
        _stub_response(
            bridge,
            [
                {
                    "entity_id": "light.office",
                    "attributes": {"friendly_name": "Office Light"},
                }
            ],
        )

        bridge._refresh_entity_cache(force=True)
        assert "office light" in bridge._entity_cache

    def test_entity_resolution_via_map(self) -> None:
        bridge = _make_bridge(entity_map={"desk lamp": "light.desk"})
        _stub_response(bridge, [])  # no HA state entities
        result = bridge._resolve_entity("desk lamp")
        assert result == "light.desk"

    def test_entity_resolution_returns_none_for_unknown(self) -> None:
        bridge = _make_bridge(entity_map={})
        _stub_response(bridge, [])
        result = bridge._resolve_entity("nonexistent device")
        assert result is None

    def test_blueprint_entities_endpoint(self) -> None:
        bridge = _make_bridge(entity_map={"tv": "media_player.tv"})
        _stub_response(
            bridge,
            [
                {
                    "entity_id": "media_player.tv",
                    "attributes": {"friendly_name": "TV"},
                }
            ],
        )

        from rex.ha_bridge import create_blueprint

        bp = create_blueprint(bridge)

        import flask

        app = flask.Flask(__name__)
        app.register_blueprint(bp)
        client = app.test_client()

        resp = client.get("/ha/entities")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "entities" in data
