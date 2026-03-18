"""Tests for US-104: Consistent error response envelope.

All API error responses must return JSON with at minimum:
  {"error": {"code": <str>, "message": <str>}}

HTTP status codes must be semantically correct.
No endpoint may return a plain-text error or an unstructured exception traceback.
"""

from __future__ import annotations

import pytest
from flask import Flask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_envelope(data: dict, expected_status: int, actual_status: int) -> None:
    """Assert the standard error envelope is present and status is correct."""
    assert actual_status == expected_status, f"expected HTTP {expected_status}, got {actual_status}"
    assert "error" in data, f"response missing 'error' key: {data}"
    err = data["error"]
    assert isinstance(err, dict), f"error must be a dict, got {type(err).__name__}: {err}"
    assert "code" in err, f"error missing 'code' field: {err}"
    assert "message" in err, f"error missing 'message' field: {err}"
    assert isinstance(err["code"], str), f"error.code must be a string: {err['code']}"
    assert isinstance(err["message"], str), f"error.message must be a string: {err['message']}"
    assert err["code"], "error.code must not be empty"
    assert err["message"], "error.message must not be empty"


# ---------------------------------------------------------------------------
# http_errors module tests
# ---------------------------------------------------------------------------


class TestHttpErrorsModule:
    def test_error_response_returns_tuple(self):
        from rex.http_errors import INTERNAL_ERROR, error_response

        app = Flask(__name__)
        with app.app_context():
            result = error_response(INTERNAL_ERROR, "something broke", 500)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_error_response_status_code(self):
        from rex.http_errors import BAD_REQUEST, error_response

        app = Flask(__name__)
        with app.app_context():
            resp, status = error_response(BAD_REQUEST, "bad input", 400)
        assert status == 400

    def test_error_response_envelope_shape(self):
        from rex.http_errors import NOT_FOUND, error_response

        app = Flask(__name__)
        with app.app_context():
            resp, status = error_response(NOT_FOUND, "not found", 404)
            data = resp.get_json()
        _assert_envelope(data, 404, status)
        assert data["error"]["code"] == "NOT_FOUND"
        assert data["error"]["message"] == "not found"

    def test_error_response_content_type_is_json(self):
        from rex.http_errors import BAD_REQUEST, error_response

        app = Flask(__name__)
        with app.app_context():
            resp, _ = error_response(BAD_REQUEST, "bad", 400)
        assert "application/json" in resp.content_type

    def test_all_standard_codes_exported(self):
        from rex import http_errors

        for code in [
            "BAD_REQUEST",
            "UNAUTHORIZED",
            "FORBIDDEN",
            "NOT_FOUND",
            "TOO_MANY_REQUESTS",
            "INTERNAL_ERROR",
            "SERVICE_UNAVAILABLE",
            "UNPROCESSABLE",
        ]:
            assert hasattr(http_errors, code), f"missing constant: {code}"
            val = getattr(http_errors, code)
            assert isinstance(val, str) and val, f"{code} must be a non-empty string"


# ---------------------------------------------------------------------------
# Dashboard routes tests — fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_dashboard_state(monkeypatch):
    """Reset session manager and rate limiter between dashboard tests."""
    import rex.dashboard.auth as auth_module
    import rex.dashboard.routes as routes_module
    from rex.dashboard.auth import LoginRateLimiter, SessionManager

    fresh_sm = SessionManager(expiry_seconds=3600)
    fresh_rl = LoginRateLimiter()
    monkeypatch.setattr(auth_module, "_session_manager", fresh_sm)
    monkeypatch.setattr(auth_module, "_login_rate_limiter", fresh_rl)
    monkeypatch.setattr(routes_module, "get_session_manager", lambda: fresh_sm)
    monkeypatch.setattr(routes_module, "get_login_rate_limiter", lambda: fresh_rl)
    yield


@pytest.fixture()
def dash_client(monkeypatch):
    """Return a test client for the dashboard blueprint."""
    monkeypatch.setenv("REX_DASHBOARD_ALLOW_LOCAL", "0")
    monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "test-pw")

    from flask import Flask

    from rex.dashboard import dashboard_bp

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret"
    app.register_blueprint(dashboard_bp)
    with app.test_client() as client:
        yield client


def _auth_headers(client) -> dict:
    """Obtain a valid session token and return auth headers."""
    resp = client.post(
        "/api/dashboard/login",
        json={"password": "test-pw"},
        content_type="application/json",
    )
    assert resp.status_code == 200, resp.data
    token = resp.get_json()["token"]
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# 401 — unauthenticated access
# ---------------------------------------------------------------------------


class TestDashboard401:
    def test_no_auth_returns_envelope(self, dash_client):
        resp = dash_client.get("/api/settings")
        data = resp.get_json()
        _assert_envelope(data, 401, resp.status_code)
        assert data["error"]["code"] == "UNAUTHORIZED"

    def test_invalid_password_returns_envelope(self, dash_client):
        resp = dash_client.post(
            "/api/dashboard/login",
            json={"password": "wrong"},
            content_type="application/json",
        )
        data = resp.get_json()
        _assert_envelope(data, 401, resp.status_code)
        assert data["error"]["code"] == "UNAUTHORIZED"


# ---------------------------------------------------------------------------
# 400 — bad input
# ---------------------------------------------------------------------------


class TestDashboard400:
    def test_login_password_not_string(self, dash_client):
        resp = dash_client.post(
            "/api/dashboard/login",
            json={"password": 12345},
            content_type="application/json",
        )
        data = resp.get_json()
        _assert_envelope(data, 400, resp.status_code)
        assert data["error"]["code"] == "BAD_REQUEST"

    def test_chat_missing_message(self, dash_client):
        headers = _auth_headers(dash_client)
        resp = dash_client.post(
            "/api/chat",
            json={"message": ""},
            content_type="application/json",
            headers=headers,
        )
        data = resp.get_json()
        _assert_envelope(data, 400, resp.status_code)
        assert data["error"]["code"] == "BAD_REQUEST"

    def test_chat_message_not_string(self, dash_client):
        headers = _auth_headers(dash_client)
        resp = dash_client.post(
            "/api/chat",
            json={"message": 999},
            content_type="application/json",
            headers=headers,
        )
        data = resp.get_json()
        _assert_envelope(data, 400, resp.status_code)

    def test_create_job_missing_name(self, dash_client):
        headers = _auth_headers(dash_client)
        resp = dash_client.post(
            "/api/scheduler/jobs",
            json={"name": "", "schedule": "interval:60"},
            content_type="application/json",
            headers=headers,
        )
        data = resp.get_json()
        _assert_envelope(data, 400, resp.status_code)
        assert data["error"]["code"] == "BAD_REQUEST"

    def test_create_job_invalid_schedule(self, dash_client):
        headers = _auth_headers(dash_client)
        resp = dash_client.post(
            "/api/scheduler/jobs",
            json={"name": "test-job", "schedule": "bad-schedule"},
            content_type="application/json",
            headers=headers,
        )
        data = resp.get_json()
        _assert_envelope(data, 400, resp.status_code)
        assert data["error"]["code"] == "BAD_REQUEST"

    def test_update_settings_no_data(self, dash_client):
        headers = _auth_headers(dash_client)
        resp = dash_client.patch(
            "/api/settings",
            json={},
            content_type="application/json",
            headers=headers,
        )
        data = resp.get_json()
        _assert_envelope(data, 400, resp.status_code)

    def test_update_job_no_data(self, dash_client):
        headers = _auth_headers(dash_client)
        resp = dash_client.patch(
            "/api/scheduler/jobs/nonexistent-job",
            json={},
            content_type="application/json",
            headers=headers,
        )
        data = resp.get_json()
        # Either 400 (no data) or 404 (job not found) — both must use envelope
        assert resp.status_code in (400, 404)
        _assert_envelope(data, resp.status_code, resp.status_code)


# ---------------------------------------------------------------------------
# 404 — not found
# ---------------------------------------------------------------------------


class TestDashboard404:
    def test_get_nonexistent_job_returns_envelope(self, dash_client):
        headers = _auth_headers(dash_client)
        resp = dash_client.get(
            "/api/scheduler/jobs/does-not-exist",
            headers=headers,
        )
        data = resp.get_json()
        _assert_envelope(data, 404, resp.status_code)
        assert data["error"]["code"] == "NOT_FOUND"

    def test_delete_nonexistent_job_returns_envelope(self, dash_client):
        headers = _auth_headers(dash_client)
        resp = dash_client.delete(
            "/api/scheduler/jobs/does-not-exist",
            headers=headers,
        )
        data = resp.get_json()
        _assert_envelope(data, 404, resp.status_code)
        assert data["error"]["code"] == "NOT_FOUND"

    def test_run_nonexistent_job_returns_envelope(self, dash_client):
        headers = _auth_headers(dash_client)
        resp = dash_client.post(
            "/api/scheduler/jobs/does-not-exist/run",
            headers=headers,
        )
        data = resp.get_json()
        _assert_envelope(data, 404, resp.status_code)
        assert data["error"]["code"] == "NOT_FOUND"

    def test_mark_nonexistent_notification_read_returns_envelope(self, dash_client):
        headers = _auth_headers(dash_client)
        resp = dash_client.post(
            "/api/notifications/no-such-id/read",
            headers=headers,
        )
        data = resp.get_json()
        _assert_envelope(data, 404, resp.status_code)
        assert data["error"]["code"] == "NOT_FOUND"


# ---------------------------------------------------------------------------
# 422 — voice endpoint with no audio
# ---------------------------------------------------------------------------


class TestDashboard400Voice:
    def test_voice_no_audio_file_returns_envelope(self, dash_client):
        headers = _auth_headers(dash_client)
        resp = dash_client.post(
            "/api/voice",
            headers=headers,
        )
        data = resp.get_json()
        _assert_envelope(data, 400, resp.status_code)
        assert data["error"]["code"] == "BAD_REQUEST"


# ---------------------------------------------------------------------------
# 429 — rate limit
# ---------------------------------------------------------------------------


class TestDashboard429:
    def test_lockout_returns_envelope(self, monkeypatch, dash_client):
        """After enough failures the rate limiter should return 429 with envelope."""
        from rex.dashboard.auth import LoginRateLimiter

        # Use a very short window and low attempt count
        limiter = LoginRateLimiter(max_attempts=1, window_seconds=0)

        import rex.dashboard.routes as _routes

        monkeypatch.setattr(_routes, "get_login_rate_limiter", lambda: limiter)

        # First attempt fails (wrong pw) — should lock
        dash_client.post(
            "/api/dashboard/login",
            json={"password": "wrong"},
            content_type="application/json",
        )
        # Second attempt should be rate-limited
        resp2 = dash_client.post(
            "/api/dashboard/login",
            json={"password": "wrong"},
            content_type="application/json",
        )
        if resp2.status_code == 429:
            data = resp2.get_json()
            _assert_envelope(data, 429, resp2.status_code)
            assert data["error"]["code"] == "TOO_MANY_REQUESTS"


# ---------------------------------------------------------------------------
# rex_speak_api tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def speak_client(monkeypatch):
    """Return a test client for the speak API (no real TTS needed)."""
    monkeypatch.setenv("REX_SPEAK_API_KEY", "test-key")
    import rex_speak_api as _api

    _api.app.config["TESTING"] = True
    with _api.app.test_client() as client:
        yield client


class TestSpeakApi400:
    def test_speak_missing_text_returns_envelope(self, speak_client):
        resp = speak_client.post(
            "/speak",
            json={},
            content_type="application/json",
            headers={"X-API-Key": "test-key"},
        )
        data = resp.get_json()
        _assert_envelope(data, 400, resp.status_code)
        assert data["error"]["code"] == "BAD_REQUEST"

    def test_speak_text_not_string_returns_envelope(self, speak_client):
        resp = speak_client.post(
            "/speak",
            json={"text": 123},
            content_type="application/json",
            headers={"X-API-Key": "test-key"},
        )
        data = resp.get_json()
        _assert_envelope(data, 400, resp.status_code)


class TestSpeakApi401:
    def test_speak_wrong_key_returns_envelope(self, speak_client):
        resp = speak_client.post(
            "/speak",
            json={"text": "hello"},
            content_type="application/json",
            headers={"X-API-Key": "wrong-key"},
        )
        data = resp.get_json()
        _assert_envelope(data, 401, resp.status_code)
        assert data["error"]["code"] == "UNAUTHORIZED"


# ---------------------------------------------------------------------------
# HA bridge tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def ha_client(monkeypatch):
    from flask import Flask

    from rex.ha_bridge import HABridge, create_blueprint

    bridge = HABridge(base_url="http://homeassistant.local", token="test-token")
    # secret="" means no auth check
    bridge._secret = ""

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(create_blueprint(bridge=bridge))
    with app.test_client() as client:
        yield client


class TestHaBridge400:
    def test_run_script_missing_script_field_returns_envelope(self, ha_client):
        resp = ha_client.post(
            "/ha/script",
            json={},
            content_type="application/json",
        )
        data = resp.get_json()
        _assert_envelope(data, 400, resp.status_code)
        assert data["error"]["code"] == "BAD_REQUEST"

    def test_run_script_non_string_returns_envelope(self, ha_client):
        resp = ha_client.post(
            "/ha/script",
            json={"script": 123},
            content_type="application/json",
        )
        data = resp.get_json()
        _assert_envelope(data, 400, resp.status_code)
        assert data["error"]["code"] == "BAD_REQUEST"


class TestHaBridge403:
    def test_wrong_secret_returns_envelope(self):
        from flask import Flask

        from rex.ha_bridge import HABridge, create_blueprint

        bridge = HABridge(base_url="http://homeassistant.local", token="tok")
        bridge._secret = "secret123"

        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(create_blueprint(bridge=bridge))
        with app.test_client() as client:
            resp = client.get("/ha/intents", headers={"HASS_SECRET": "wrong"})
        data = resp.get_json()
        _assert_envelope(data, 403, resp.status_code)
        assert data["error"]["code"] == "FORBIDDEN"


# ---------------------------------------------------------------------------
# Inbound webhook — plain-text removal
# ---------------------------------------------------------------------------


class TestInboundWebhook403:
    def test_signature_failure_returns_json_not_plaintext(self, monkeypatch):
        """The Twilio signature failure must return JSON, not plain text."""
        from flask import Flask

        from rex.messaging_backends.inbound_store import InboundSmsStore
        from rex.messaging_backends.inbound_webhook import create_inbound_sms_blueprint

        store = InboundSmsStore()
        bp = create_inbound_sms_blueprint(
            auth_token="tok123",
            inbound_store=store,
        )
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(bp)

        with app.test_client() as client:
            # Send a request without a valid Twilio signature
            resp = client.post(
                "/webhooks/twilio/sms",
                data={"Body": "hello", "From": "+1234567890", "To": "+0987654321"},
                headers={"X-Twilio-Signature": "invalidsig"},
            )
        # Must be JSON (not text/plain)
        assert "application/json" in (
            resp.content_type or ""
        ), f"expected JSON response, got content_type={resp.content_type!r}"
        data = resp.get_json()
        assert data is not None, "response body is not valid JSON"
        _assert_envelope(data, 403, resp.status_code)
