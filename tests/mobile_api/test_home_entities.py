from __future__ import annotations

from unittest.mock import patch

from tests.mobile_api.conftest import auth_header, create_user, paired_login_tokens


def _headers(client, *, admin: bool = True, scopes: list[str] | None = None):
    create_user("james", "correct-horse", admin=admin)
    tokens = paired_login_tokens(
        client,
        "james",
        "correct-horse",
        scopes=scopes or ["home.read"],
    )
    return auth_header(tokens["access_token"])


def test_home_entities_projects_only_supported_devices(client):
    headers = _headers(client)
    raw = [
        {
            "entity_id": "light.kitchen",
            "friendly_name": "Kitchen Lights",
            "domain": "light",
            "state": "on",
            "area_name": "Kitchen",
        },
        {
            "entity_id": "cover.garage_door",
            "friendly_name": "Garage Door",
            "domain": "cover",
            "state": "closed",
            "area_name": "Garage",
        },
        {
            "entity_id": "climate.downstairs",
            "friendly_name": "Downstairs Thermostat",
            "domain": "climate",
            "state": "heat",
        },
        {
            "entity_id": "cover.living_room_shade",
            "friendly_name": "Living Room Shade",
            "domain": "cover",
            "state": "open",
        },
        {
            "entity_id": "sensor.private_temperature",
            "friendly_name": "Private Temperature",
            "domain": "sensor",
            "state": "72",
        },
        {
            "entity_id": "light.mismatch",
            "friendly_name": "Invalid",
            "domain": "switch",
            "state": "on",
        },
    ]
    with patch("rex.mobile_api.routes.home._load_home_entities", return_value=(True, raw)):
        response = client.get("/mobile/home/entities", headers=headers)
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["configured"] is True
    assert [device["id"] for device in body["devices"]] == [
        "cover.garage_door",
        "climate.downstairs",
        "light.kitchen",
    ]
    by_id = {device["id"]: device for device in body["devices"]}
    assert by_id["light.kitchen"] == {
        "id": "light.kitchen",
        "name": "Kitchen Lights",
        "type": "light",
        "room": "kitchen",
        "state": "on",
        "riskLevel": "low",
    }
    assert by_id["cover.garage_door"]["type"] == "garage"
    assert by_id["cover.garage_door"]["riskLevel"] == "high"
    assert by_id["climate.downstairs"]["riskLevel"] == "medium"
    assert body["rooms"][0] == {
        "id": "all",
        "name": "All",
        "deviceCount": 3,
        "activeCount": 1,
    }


def test_home_entities_not_configured_is_truthful_and_empty(client):
    headers = _headers(client)
    with patch("rex.mobile_api.routes.home._load_home_entities", return_value=(False, [])):
        response = client.get("/mobile/home/entities", headers=headers)
    assert response.status_code == 200
    assert response.get_json() == {
        "configured": False,
        "devices": [],
        "rooms": [
            {
                "id": "all",
                "name": "All",
                "deviceCount": 0,
                "activeCount": 0,
            }
        ],
    }


def test_home_entities_requires_device_home_read_scope(client):
    headers = _headers(client, scopes=["chat.send"])
    response = client.get("/mobile/home/entities", headers=headers)
    assert response.status_code == 403
    assert response.get_json()["error"]["code"] == "FORBIDDEN"


def test_home_entities_requires_live_user_ha_permission(client):
    headers = _headers(client, admin=False)
    response = client.get("/mobile/home/entities", headers=headers)
    assert response.status_code == 403
    assert response.get_json()["error"]["code"] == "FORBIDDEN"


def test_home_entities_backend_failure_is_retryable_and_secret_free(client):
    headers = _headers(client)
    with patch(
        "rex.mobile_api.routes.home._load_home_entities",
        side_effect=RuntimeError("Bearer secret-token"),
    ):
        response = client.get("/mobile/home/entities", headers=headers)
    assert response.status_code == 503
    error = response.get_json()["error"]
    assert error["code"] == "BACKEND_UNAVAILABLE"
    assert error["retryable"] is True
    assert "secret-token" not in error["message"]
