"""Tests for US-050: Web UI server.

Acceptance criteria:
- UI server starts
- UI accessible
- interface renders
- Typecheck passes
- Verify changes work in browser
"""

from __future__ import annotations

import pytest
from flask import Flask

from rex.dashboard import dashboard_bp


@pytest.fixture()
def app():
    """Create a minimal Flask app with the dashboard blueprint registered."""
    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.config["SECRET_KEY"] = "test-secret-ui"
    flask_app.register_blueprint(dashboard_bp)
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


# --- UI server starts ---


def test_web_ui_blueprint_registers():
    """Dashboard blueprint registers without error on a fresh Flask app."""
    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.register_blueprint(dashboard_bp)
    rules = {rule.rule for rule in flask_app.url_map.iter_rules()}
    assert "/dashboard" in rules


def test_web_ui_has_required_routes(app):
    """Dashboard registers the /dashboard and /dashboard/assets routes."""
    rules = {rule.rule for rule in app.url_map.iter_rules()}
    assert "/dashboard" in rules
    assert "/dashboard/assets/<path:filename>" in rules


def test_web_ui_server_starts_with_test_client(app):
    """Flask test client can be created for the app (server is usable)."""
    with app.test_client() as c:
        response = c.get("/api/dashboard/status")
    assert response.status_code == 200


# --- UI accessible ---


def test_dashboard_route_returns_200(client):
    """GET /dashboard returns HTTP 200."""
    response = client.get("/dashboard")
    assert response.status_code == 200


def test_dashboard_route_returns_html(client):
    """GET /dashboard returns HTML content."""
    response = client.get("/dashboard")
    content_type = response.content_type
    assert "html" in content_type or response.data.startswith(b"<!DOCTYPE")


def test_dashboard_ui_response_not_empty(client):
    """GET /dashboard returns a non-empty body."""
    response = client.get("/dashboard")
    assert len(response.data) > 0


def test_notifications_ui_route_accessible(client):
    """GET /dashboard/notifications returns HTTP 200."""
    response = client.get("/dashboard/notifications")
    assert response.status_code == 200


# --- interface renders ---


def test_dashboard_html_contains_doctype(client):
    """Dashboard HTML starts with a DOCTYPE declaration."""
    response = client.get("/dashboard")
    html = response.data.decode("utf-8")
    assert "<!DOCTYPE html>" in html or "<!doctype html>" in html.lower()


def test_dashboard_html_has_title(client):
    """Dashboard HTML includes a <title> element."""
    response = client.get("/dashboard")
    html = response.data.decode("utf-8")
    assert "<title>" in html


def test_dashboard_html_title_contains_rex(client):
    """Dashboard title mentions Rex."""
    response = client.get("/dashboard")
    html = response.data.decode("utf-8")
    assert "Rex" in html


def test_dashboard_html_has_app_div(client):
    """Dashboard HTML has the root #app div."""
    response = client.get("/dashboard")
    html = response.data.decode("utf-8")
    assert 'id="app"' in html


def test_dashboard_html_has_login_form(client):
    """Dashboard HTML includes a login form."""
    response = client.get("/dashboard")
    html = response.data.decode("utf-8")
    assert "login" in html.lower()
    assert "<form" in html


def test_dashboard_html_has_css_link(client):
    """Dashboard HTML links to a stylesheet."""
    response = client.get("/dashboard")
    html = response.data.decode("utf-8")
    assert "<link" in html
    assert ".css" in html


def test_dashboard_html_has_nav_links(client):
    """Dashboard HTML contains navigation links (Chat, Settings, etc.)."""
    response = client.get("/dashboard")
    html = response.data.decode("utf-8")
    assert "Chat" in html
    assert "Settings" in html


def test_dashboard_static_css_accessible(client):
    """CSS file is served at /dashboard/assets/css/dashboard.css."""
    response = client.get("/dashboard/assets/css/dashboard.css")
    assert response.status_code == 200


def test_dashboard_static_js_accessible(client):
    """JS file is served at /dashboard/assets/js/dashboard.js."""
    response = client.get("/dashboard/assets/js/dashboard.js")
    assert response.status_code == 200


def test_dashboard_static_css_content_type(client):
    """CSS file is served with text/css content type."""
    response = client.get("/dashboard/assets/css/dashboard.css")
    assert "css" in response.content_type or "text" in response.content_type


def test_dashboard_static_js_not_empty(client):
    """JS file has non-empty content."""
    response = client.get("/dashboard/assets/js/dashboard.js")
    assert len(response.data) > 0


# --- Verify changes work in browser ---


def test_dashboard_server_serves_full_html_page(client):
    """Dashboard returns a complete HTML page with all required structural elements."""
    response = client.get("/dashboard")
    html = response.data.decode("utf-8")

    # Full page structure checks
    assert "<!DOCTYPE html>" in html or "<!doctype html>" in html.lower()
    assert "<html" in html
    assert "<head>" in html or "<head " in html
    assert "<body>" in html or "<body " in html
    assert "</html>" in html


def test_dashboard_html_has_viewport_meta(client):
    """Dashboard HTML has a viewport meta tag (mobile-responsive)."""
    response = client.get("/dashboard")
    html = response.data.decode("utf-8")
    assert 'name="viewport"' in html


def test_dashboard_api_and_ui_both_accessible(client):
    """Both the UI and the status API are accessible on the same server."""
    ui_response = client.get("/dashboard")
    api_response = client.get("/api/dashboard/status")
    assert ui_response.status_code == 200
    assert api_response.status_code == 200
    api_data = api_response.get_json()
    assert api_data["status"] == "ok"
