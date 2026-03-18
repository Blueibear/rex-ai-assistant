"""Tests for US-119: Rate limiting on public-facing API endpoints.

Acceptance criteria:
- rate limiter applied to all unauthenticated or public endpoints
- rate limit configurable (default: 60 requests/minute per IP)
- requests exceeding the limit receive a 429 response with a Retry-After header
- rate limiter does not apply to health check endpoints
- Typecheck passes
"""

from __future__ import annotations

from typing import Any

import pytest
from flask import Flask

from rex.rate_limiter import install_rate_limiter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app() -> Flask:
    """Minimal Flask app with a very low rate limit (3/minute) for testing."""
    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True

    # Use a low limit so tests can exhaust it quickly.
    install_rate_limiter(flask_app, limit_str="3 per minute")

    @flask_app.route("/public")
    def public_endpoint() -> Any:
        return {"ok": True}, 200

    @flask_app.route("/health/live")
    def health_live() -> Any:
        return {"status": "ok"}, 200

    @flask_app.route("/health/ready")
    def health_ready() -> Any:
        return {"status": "ok"}, 200

    return flask_app


@pytest.fixture()
def client(app: Flask):  # type: ignore[type-arg]
    return app.test_client()


# ---------------------------------------------------------------------------
# install_rate_limiter unit tests
# ---------------------------------------------------------------------------


class TestInstallRateLimiter:
    def test_idempotent_double_install(self) -> None:
        flask_app = Flask(__name__)
        flask_app.config["TESTING"] = True
        limiter1 = install_rate_limiter(flask_app, limit_str="60 per minute")
        limiter2 = install_rate_limiter(flask_app, limit_str="60 per minute")
        assert limiter1 is limiter2

    def test_registered_in_extensions(self) -> None:
        flask_app = Flask(__name__)
        flask_app.config["TESTING"] = True
        install_rate_limiter(flask_app, limit_str="60 per minute")
        assert flask_app.extensions.get("rex_rate_limiter_installed") is not None


# ---------------------------------------------------------------------------
# Rate limit enforcement
# ---------------------------------------------------------------------------


class TestRateLimitEnforcement:
    def test_requests_within_limit_succeed(self, client: Any) -> None:
        """First N requests within the limit must all succeed."""
        for _ in range(3):
            resp = client.get("/public")
            assert resp.status_code == 200

    def test_exceeding_limit_returns_429(self, client: Any) -> None:
        """One request past the limit must return 429."""
        for _ in range(3):
            client.get("/public")
        resp = client.get("/public")  # 4th request — over limit
        assert resp.status_code == 429

    def test_429_returns_error_envelope(self, client: Any) -> None:
        for _ in range(3):
            client.get("/public")
        resp = client.get("/public")
        data = resp.get_json()
        assert "error" in data
        assert data["error"]["code"] == "TOO_MANY_REQUESTS"
        assert "message" in data["error"]

    def test_429_has_retry_after_header(self, client: Any) -> None:
        for _ in range(3):
            client.get("/public")
        resp = client.get("/public")
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers

    def test_retry_after_is_positive_integer(self, client: Any) -> None:
        for _ in range(3):
            client.get("/public")
        resp = client.get("/public")
        retry_after = resp.headers.get("Retry-After", "")
        assert retry_after.isdigit(), f"Retry-After must be a number, got {retry_after!r}"
        assert int(retry_after) >= 1


# ---------------------------------------------------------------------------
# Health check endpoints are exempt
# ---------------------------------------------------------------------------


class TestHealthEndpointsExempt:
    def test_health_live_not_rate_limited(self, client: Any) -> None:
        """Health live endpoint must succeed even after limit is exhausted."""
        for _ in range(3):
            client.get("/public")
        # These must still return 200 regardless of the rate limit.
        for _ in range(5):
            resp = client.get("/health/live")
            assert (
                resp.status_code == 200
            ), f"Health endpoint returned {resp.status_code} but should be exempt"

    def test_health_ready_not_rate_limited(self, client: Any) -> None:
        for _ in range(3):
            client.get("/public")
        for _ in range(5):
            resp = client.get("/health/ready")
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Configurable limit from environment variable
# ---------------------------------------------------------------------------


class TestRateLimitConfiguration:
    def test_default_limit_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """install_rate_limiter reads API_RATE_LIMIT env var when limit_str is None."""
        monkeypatch.setenv("API_RATE_LIMIT", "2 per minute")

        flask_app = Flask(__name__)
        flask_app.config["TESTING"] = True
        install_rate_limiter(flask_app)  # no explicit limit_str

        @flask_app.route("/ep")
        def _ep() -> Any:
            return {}, 200

        with flask_app.test_client() as c:
            c.get("/ep")
            c.get("/ep")
            resp = c.get("/ep")  # 3rd request — over the env-var limit of 2
        assert resp.status_code == 429

    def test_default_limit_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When API_RATE_LIMIT is unset, fall back to 60 per minute."""
        monkeypatch.delenv("API_RATE_LIMIT", raising=False)

        from rex.rate_limiter import _DEFAULT_LIMIT, _read_limit

        assert _read_limit() == _DEFAULT_LIMIT

    def test_env_override_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("API_RATE_LIMIT", "100 per hour")
        from rex.rate_limiter import _read_limit

        assert _read_limit() == "100 per hour"

    def test_empty_env_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("API_RATE_LIMIT", "")
        from rex.rate_limiter import _DEFAULT_LIMIT, _read_limit

        assert _read_limit() == _DEFAULT_LIMIT

    def test_explicit_limit_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit limit_str takes priority over the environment variable."""
        monkeypatch.setenv("API_RATE_LIMIT", "10 per minute")

        flask_app = Flask(__name__)
        flask_app.config["TESTING"] = True
        install_rate_limiter(flask_app, limit_str="1 per minute")

        @flask_app.route("/x")
        def _x() -> Any:
            return {}, 200

        with flask_app.test_client() as c:
            c.get("/x")
            resp = c.get("/x")  # 2nd request — over explicit limit of 1
        assert resp.status_code == 429
