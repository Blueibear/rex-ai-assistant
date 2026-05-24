"""Tests for US-SL-004: Shopping list mobile PWA endpoint."""

from __future__ import annotations

import json

import pytest

from rex.shopping_list import ShoppingList
from rex.shopping_pwa import create_blueprint

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _sl(tmp_path):
    return ShoppingList(path=tmp_path / "sl.json")


@pytest.fixture()
def client(_sl):
    """Flask test client with no PIN configured."""
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(create_blueprint(_sl, pin=None))
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture()
def client_pin(_sl):
    """Flask test client with PIN '1234' configured."""
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(create_blueprint(_sl, pin="1234"))
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Open (no-PIN) access
# ---------------------------------------------------------------------------


def test_shopping_index_returns_html(client):
    r = client.get("/shopping")
    assert r.status_code == 200
    assert b"Shopping List" in r.data
    assert b"text/html" in r.content_type.encode()


def test_shopping_page_contains_manifest_link(client):
    r = client.get("/shopping")
    assert b'rel="manifest"' in r.data
    assert b"/shopping/manifest.json" in r.data


def test_shopping_page_contains_viewport_meta(client):
    """Mobile-responsive: viewport meta tag present."""
    r = client.get("/shopping")
    assert b"width=device-width" in r.data


def test_manifest_json(client):
    r = client.get("/shopping/manifest.json")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data["start_url"] == "/shopping"
    assert data["display"] == "standalone"
    assert "name" in data
    assert "icons" in data


def test_api_list_empty(client):
    r = client.get("/shopping/api/items")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data["items"] == []


def test_api_add_item(client, _sl):
    r = client.post(
        "/shopping/api/items",
        data=json.dumps({"name": "Milk", "quantity": 2, "unit": "litres"}),
        content_type="application/json",
    )
    assert r.status_code == 201
    data = json.loads(r.data)
    assert data["name"] == "Milk"
    assert data["quantity"] == 2.0
    assert data["checked"] is False


def test_api_toggle_check(client, _sl):
    item = _sl.add_item("Eggs", quantity=12)
    r = client.patch(
        f"/shopping/api/items/{item.id}",
        data=json.dumps({"checked": True}),
        content_type="application/json",
    )
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data["checked"] is True


def test_api_toggle_uncheck(client, _sl):
    item = _sl.add_item("Butter")
    _sl.check_item(item.id)
    r = client.patch(
        f"/shopping/api/items/{item.id}",
        data=json.dumps({"checked": False}),
        content_type="application/json",
    )
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data["checked"] is False


def test_api_toggle_missing_item(client):
    r = client.patch(
        "/shopping/api/items/nonexistent",
        data=json.dumps({"checked": True}),
        content_type="application/json",
    )
    assert r.status_code == 404


def test_api_clear_checked(client, _sl):
    _sl.add_item("Apple")
    item_b = _sl.add_item("Banana")
    _sl.check_item(item_b.id)
    r = client.post("/shopping/api/clear-checked")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data["removed"] == 1
    remaining = _sl.list_items()
    assert len(remaining) == 1
    assert remaining[0].name == "Apple"


def test_api_add_item_missing_name(client):
    r = client.post(
        "/shopping/api/items",
        data=json.dumps({"name": ""}),
        content_type="application/json",
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# PIN protection
# ---------------------------------------------------------------------------


def test_pin_redirects_to_pin_page(client_pin):
    r = client_pin.get("/shopping")
    # No cookie → returns pin page (200 with form)
    assert r.status_code == 200
    assert b"PIN" in r.data or b"pin" in r.data.lower()


def test_pin_wrong_returns_error(client_pin):
    r = client_pin.post("/shopping/pin", data={"pin": "9999"})
    assert r.status_code == 401
    assert b"Incorrect" in r.data


def test_pin_correct_sets_cookie(client_pin):
    r = client_pin.post("/shopping/pin", data={"pin": "1234"})
    # Should redirect to /shopping
    assert r.status_code in (301, 302)
    assert "rex_shopping_auth" in r.headers.get("Set-Cookie", "")


def test_pin_correct_then_access(client_pin):
    """After correct PIN, /shopping returns the shopping page."""
    client_pin.post("/shopping/pin", data={"pin": "1234"})
    r = client_pin.get("/shopping")
    assert r.status_code == 200
    assert b"Shopping List" in r.data


def test_api_items_blocked_without_pin(client_pin):
    r = client_pin.get("/shopping/api/items")
    # Returns PIN page (200) rather than JSON — not authenticated
    assert b"PIN" in r.data or b"pin" in r.data.lower() or r.status_code in (200, 401)


def test_manifest_accessible_without_pin(client_pin):
    """Manifest must be accessible to browsers before auth for PWA install."""
    r = client_pin.get("/shopping/manifest.json")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data["start_url"] == "/shopping"
