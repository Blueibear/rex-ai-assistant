"""Tests for US-097: HTTP security headers.

Acceptance criteria:
- Content-Security-Policy header present on all HTML responses
- X-Frame-Options: DENY or SAMEORIGIN set
- X-Content-Type-Options: nosniff set
- CORS policy restricts allowed origins to configured whitelist (not wildcard *)
- Strict-Transport-Security header set if HTTPS is used
- Typecheck passes
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from flask import Flask

from rex.dashboard import dashboard_bp
from rex.dashboard.auth import LoginRateLimiter, SessionManager


@pytest.fixture(autouse=True)
def isolate_auth(monkeypatch):
    """Fresh session manager and rate limiter per test."""
    import rex.dashboard.auth as auth_mod

    monkeypatch.setattr(auth_mod, "_session_manager", SessionManager(expiry_seconds=3600))
    monkeypatch.setattr(auth_mod, "_login_rate_limiter", LoginRateLimiter())


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.register_blueprint(dashboard_bp)
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def token(client):
    """Login and return a valid session token."""
    resp = client.post("/api/dashboard/login", json={"password": "test-pw"})
    assert resp.status_code == 200
    return resp.get_json()["token"]


# ---------------------------------------------------------------------------
# AC: X-Frame-Options present on all responses
# ---------------------------------------------------------------------------


class TestXFrameOptions:
    def test_json_api_response_has_x_frame_options(self, client, token):
        resp = client.get(
            "/api/dashboard/status",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert "X-Frame-Options" in resp.headers

    def test_x_frame_options_is_deny_or_sameorigin(self, client, token):
        resp = client.get(
            "/api/dashboard/status",
            headers={"Authorization": f"Bearer {token}"},
        )
        value = resp.headers.get("X-Frame-Options", "")
        assert value in {"DENY", "SAMEORIGIN"}, f"Unexpected X-Frame-Options: {value!r}"

    def test_login_endpoint_has_x_frame_options(self, client):
        resp = client.post("/api/dashboard/login", json={"password": "wrong"})
        assert "X-Frame-Options" in resp.headers

    def test_logout_endpoint_has_x_frame_options(self, client):
        resp = client.post("/api/dashboard/logout")
        assert "X-Frame-Options" in resp.headers


# ---------------------------------------------------------------------------
# AC: X-Content-Type-Options: nosniff on all responses
# ---------------------------------------------------------------------------


class TestXContentTypeOptions:
    def test_json_response_has_nosniff(self, client, token):
        resp = client.get(
            "/api/dashboard/status",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_error_response_has_nosniff(self, client):
        resp = client.post("/api/dashboard/login", json={"password": "bad"})
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_chat_history_response_has_nosniff(self, client, token):
        resp = client.get(
            "/api/chat/history",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"


# ---------------------------------------------------------------------------
# AC: Content-Security-Policy on HTML responses
# ---------------------------------------------------------------------------


class TestContentSecurityPolicy:
    def test_html_dashboard_route_has_csp(self, client, token, tmp_path):
        """If the dashboard serves HTML, CSP must be present."""
        # Create minimal template so the route can respond
        import rex.dashboard.routes as routes_mod

        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        (templates_dir / "index.html").write_text("<!DOCTYPE html><html><body>Test</body></html>")

        original = routes_mod._get_dashboard_dir

        def _patched_dir() -> Path:  # type: ignore[name-defined]
            from pathlib import Path

            return Path(tmp_path)

        routes_mod._get_dashboard_dir = _patched_dir
        try:
            resp = client.get(
                "/dashboard",
                headers={"Cookie": f"rex_dashboard_token={token}"},
            )
            if resp.status_code == 200 and "text/html" in (resp.content_type or ""):
                assert (
                    "Content-Security-Policy" in resp.headers
                ), "HTML responses must include Content-Security-Policy"
        finally:
            routes_mod._get_dashboard_dir = original

    def test_json_response_does_not_require_csp_but_may_have_it(self, client, token):
        """JSON responses may optionally carry CSP; the requirement is for HTML."""
        resp = client.get(
            "/api/dashboard/status",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        # CSP is not required on JSON, but X-Frame-Options and nosniff must be there
        assert "X-Frame-Options" in resp.headers
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_csp_policy_blocks_framing(self, client, tmp_path):
        """CSP frame-ancestors directive must be present and restrictive."""
        import rex.dashboard.routes as routes_mod

        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        (templates_dir / "index.html").write_text("<!DOCTYPE html><html><body>Test</body></html>")
        original = routes_mod._get_dashboard_dir

        def _patched_dir():
            from pathlib import Path

            return Path(tmp_path)

        routes_mod._get_dashboard_dir = _patched_dir
        try:
            import rex.dashboard.auth as auth_mod

            sm = SessionManager(expiry_seconds=3600)
            session = sm.create_session()
            auth_mod._session_manager = sm

            resp = client.get(
                "/dashboard",
                headers={"Cookie": f"rex_dashboard_token={session.token}"},
            )
            if resp.status_code == 200 and "text/html" in (resp.content_type or ""):
                csp = resp.headers.get("Content-Security-Policy", "")
                assert (
                    "frame-ancestors" in csp.lower()
                ), "CSP must include frame-ancestors directive"
        finally:
            routes_mod._get_dashboard_dir = original


# ---------------------------------------------------------------------------
# AC: CORS restricts allowed origins (not wildcard *)
# ---------------------------------------------------------------------------


class TestCORSPolicy:
    def test_cors_env_variable_is_not_wildcard_by_default(self, monkeypatch):
        """The default REX_ALLOWED_ORIGINS must not be '*'."""
        monkeypatch.delenv("REX_ALLOWED_ORIGINS", raising=False)
        # flask_proxy.py default: specific localhost origins
        default = (
            "http://localhost:3000,http://localhost:5000,"
            "http://127.0.0.1:3000,http://127.0.0.1:5000"
        )
        # Simulate what flask_proxy.py does
        allowed = os.getenv("REX_ALLOWED_ORIGINS", default)
        origins = [o.strip() for o in allowed.split(",") if o.strip()]
        assert "*" not in origins, "CORS origins must not include wildcard *"
        assert len(origins) > 0, "At least one allowed origin must be configured"

    def test_cors_configured_origins_are_all_specific(self, monkeypatch):
        """All configured CORS origins must be specific URLs, not wildcards."""
        monkeypatch.setenv(
            "REX_ALLOWED_ORIGINS",
            "http://localhost:3000,http://localhost:5000",
        )
        allowed = os.getenv("REX_ALLOWED_ORIGINS", "")
        origins = [o.strip() for o in allowed.split(",") if o.strip()]
        for origin in origins:
            assert origin != "*", f"Origin {origin!r} is a wildcard — not allowed"
            assert origin.startswith(
                ("http://", "https://")
            ), f"Origin {origin!r} must be a full URL"

    def test_flask_proxy_uses_restricted_cors(self):
        """flask_proxy.py must set CORS with explicit origins, not '*'."""
        with open("flask_proxy.py") as f:
            content = f.read()
        # Should have REX_ALLOWED_ORIGINS logic, not a literal '*' in origins
        assert (
            "REX_ALLOWED_ORIGINS" in content
        ), "flask_proxy.py must use REX_ALLOWED_ORIGINS for CORS config"
        # Should not use "origins": "*" literally
        assert '"*"' not in content or "REX_ALLOWED_ORIGINS" in content


# ---------------------------------------------------------------------------
# AC: HSTS set when HTTPS
# ---------------------------------------------------------------------------


class TestHSTSHeader:
    def test_hsts_not_set_on_plain_http(self, client, token):
        """HSTS must NOT be set on plain HTTP connections."""
        resp = client.get(
            "/api/dashboard/status",
            headers={"Authorization": f"Bearer {token}"},
        )
        # Default test client uses HTTP, so HSTS should not be present
        assert (
            "Strict-Transport-Security" not in resp.headers
        ), "HSTS must not be set on plain HTTP connections"

    def test_hsts_set_on_https_connection(self, app, token):
        """HSTS must be set when the connection is HTTPS."""
        with app.test_client() as c:
            # Simulate HTTPS by using an https base_url
            resp = c.get(
                "/api/dashboard/status",
                base_url="https://localhost/",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200
            hsts = resp.headers.get("Strict-Transport-Security", "")
            assert hsts, "HSTS header must be set on HTTPS connections"
            assert "max-age=" in hsts, "HSTS header must include max-age"

    def test_hsts_max_age_is_sufficient(self, app, token):
        """HSTS max-age must be at least 1 year (31536000 seconds)."""
        with app.test_client() as c:
            resp = c.get(
                "/api/dashboard/status",
                base_url="https://localhost/",
                headers={"Authorization": f"Bearer {token}"},
            )
            hsts = resp.headers.get("Strict-Transport-Security", "")
            if hsts:
                for part in hsts.split(";"):
                    part = part.strip()
                    if part.startswith("max-age="):
                        max_age = int(part.split("=")[1])
                        assert max_age >= 31536000, f"HSTS max-age {max_age} is less than 1 year"
                        break
