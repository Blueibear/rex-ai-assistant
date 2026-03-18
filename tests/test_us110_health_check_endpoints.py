"""Tests for US-110: Liveness and readiness health check endpoints.

Verifies:
- GET /health/live returns 200 regardless of dependency state
- GET /health/ready returns 200 when all checks pass
- GET /health/ready returns 503 with JSON body when a check fails
- Response JSON describes unavailable dependencies
- Built-in check_config and check_dashboard_db helpers work correctly
- Both endpoints respond quickly (< 500ms under normal conditions)
"""

from __future__ import annotations

import time
from unittest.mock import patch

from flask import Flask

from rex.health import (
    ReadinessCheck,
    check_config,
    check_dashboard_db,
    create_health_blueprint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(checks: list[ReadinessCheck] | None = None) -> Flask:
    app = Flask(__name__)
    app.config["TESTING"] = True
    bp = create_health_blueprint(checks=checks)
    app.register_blueprint(bp)
    return app


# ---------------------------------------------------------------------------
# Liveness tests
# ---------------------------------------------------------------------------


class TestLiveness:
    def test_live_returns_200(self) -> None:
        app = _make_app()
        with app.test_client() as client:
            resp = client.get("/health/live")
        assert resp.status_code == 200

    def test_live_returns_json(self) -> None:
        app = _make_app()
        with app.test_client() as client:
            resp = client.get("/health/live")
        data = resp.get_json()
        assert isinstance(data, dict)
        assert data.get("status") == "ok"

    def test_live_returns_200_even_when_checks_fail(self) -> None:
        """Liveness is independent of readiness checks."""

        def _always_fail() -> str:
            return "dependency is down"

        app = _make_app(checks=[_always_fail])
        with app.test_client() as client:
            resp = client.get("/health/live")
        assert resp.status_code == 200

    def test_live_content_type_json(self) -> None:
        app = _make_app()
        with app.test_client() as client:
            resp = client.get("/health/live")
        assert "application/json" in resp.content_type

    def test_live_responds_quickly(self) -> None:
        """Liveness must respond in under 500ms."""
        app = _make_app()
        with app.test_client() as client:
            t0 = time.monotonic()
            client.get("/health/live")
            elapsed = time.monotonic() - t0
        assert elapsed < 0.5


# ---------------------------------------------------------------------------
# Readiness tests — all checks pass
# ---------------------------------------------------------------------------


class TestReadinessPass:
    def test_ready_returns_200_no_checks(self) -> None:
        app = _make_app(checks=[])
        with app.test_client() as client:
            resp = client.get("/health/ready")
        assert resp.status_code == 200

    def test_ready_returns_200_with_passing_check(self) -> None:
        def _ok() -> None:
            return None

        app = _make_app(checks=[_ok])
        with app.test_client() as client:
            resp = client.get("/health/ready")
        assert resp.status_code == 200

    def test_ready_json_status_ready(self) -> None:
        app = _make_app(checks=[])
        with app.test_client() as client:
            resp = client.get("/health/ready")
        data = resp.get_json()
        assert data is not None
        assert data.get("status") == "ready"

    def test_ready_content_type_json(self) -> None:
        app = _make_app(checks=[])
        with app.test_client() as client:
            resp = client.get("/health/ready")
        assert "application/json" in resp.content_type

    def test_ready_responds_quickly(self) -> None:
        """Readiness must respond in under 500ms (no real deps in test)."""
        app = _make_app(checks=[])
        with app.test_client() as client:
            t0 = time.monotonic()
            client.get("/health/ready")
            elapsed = time.monotonic() - t0
        assert elapsed < 0.5

    def test_ready_multiple_passing_checks(self) -> None:
        def _check_a() -> None:
            return None

        def _check_b() -> None:
            return None

        app = _make_app(checks=[_check_a, _check_b])
        with app.test_client() as client:
            resp = client.get("/health/ready")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Readiness tests — checks fail
# ---------------------------------------------------------------------------


class TestReadinessFail:
    def test_ready_returns_503_when_check_returns_string(self) -> None:
        def _bad() -> str:
            return "database is down"

        app = _make_app(checks=[_bad])
        with app.test_client() as client:
            resp = client.get("/health/ready")
        assert resp.status_code == 503

    def test_ready_json_contains_unavailable_key(self) -> None:
        def _bad() -> str:
            return "cannot connect"

        app = _make_app(checks=[_bad])
        with app.test_client() as client:
            resp = client.get("/health/ready")
        data = resp.get_json()
        assert data is not None
        assert "unavailable" in data

    def test_ready_json_status_unavailable(self) -> None:
        def _bad() -> str:
            return "connection refused"

        app = _make_app(checks=[_bad])
        with app.test_client() as client:
            resp = client.get("/health/ready")
        data = resp.get_json()
        assert data is not None
        assert data.get("status") == "unavailable"

    def test_ready_unavailable_contains_check_name(self) -> None:
        def my_custom_check() -> str:
            return "custom failure"

        app = _make_app(checks=[my_custom_check])
        with app.test_client() as client:
            resp = client.get("/health/ready")
        data = resp.get_json()
        assert data is not None
        assert "my_custom_check" in data.get("unavailable", {})

    def test_ready_unavailable_contains_failure_reason(self) -> None:
        def bad_db() -> str:
            return "timeout after 5s"

        app = _make_app(checks=[bad_db])
        with app.test_client() as client:
            resp = client.get("/health/ready")
        data = resp.get_json()
        assert "timeout after 5s" in data.get("unavailable", {}).get("bad_db", "")

    def test_ready_503_when_check_raises(self) -> None:
        def _raises() -> None:
            raise RuntimeError("unexpected failure")

        app = _make_app(checks=[_raises])
        with app.test_client() as client:
            resp = client.get("/health/ready")
        assert resp.status_code == 503

    def test_ready_partial_fail(self) -> None:
        """One passing + one failing → 503 with only the failing check listed."""

        def _ok() -> None:
            return None

        def _bad() -> str:
            return "redis down"

        app = _make_app(checks=[_ok, _bad])
        with app.test_client() as client:
            resp = client.get("/health/ready")
        assert resp.status_code == 503
        data = resp.get_json()
        assert "_ok" not in data.get("unavailable", {})
        assert "_bad" in data.get("unavailable", {})


# ---------------------------------------------------------------------------
# Built-in check helpers
# ---------------------------------------------------------------------------


class TestCheckConfig:
    def test_returns_none_when_config_loads(self) -> None:
        with patch("rex.config.load_config", return_value={}):
            result = check_config()
        assert result is None

    def test_returns_string_when_config_raises(self) -> None:
        with patch("rex.config.load_config", side_effect=FileNotFoundError("missing")):
            result = check_config()
        assert result is not None
        assert isinstance(result, str)
        assert "config" in result.lower() or "missing" in result.lower()


class TestCheckDashboardDb:
    def test_returns_none_when_db_accessible(self) -> None:
        mock_store = type("S", (), {"query_recent": lambda self, limit: []})()
        with patch("rex.dashboard_store.DashboardStore", return_value=mock_store):
            result = check_dashboard_db()
        assert result is None

    def test_returns_string_when_db_raises(self) -> None:
        class _FailStore:
            def query_recent(self, limit: int) -> None:  # type: ignore[return]
                raise ConnectionError("db unreachable")

        with patch("rex.dashboard_store.DashboardStore", return_value=_FailStore()):
            result = check_dashboard_db()
        assert result is not None
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Blueprint registration
# ---------------------------------------------------------------------------


class TestBlueprintRegistration:
    def test_health_endpoints_accessible_after_register(self) -> None:
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(create_health_blueprint())
        with app.test_client() as client:
            assert client.get("/health/live").status_code == 200
            assert client.get("/health/ready").status_code == 200

    def test_url_prefix_applied(self) -> None:
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(create_health_blueprint(url_prefix="/api/v1"))
        with app.test_client() as client:
            assert client.get("/api/v1/health/live").status_code == 200

    def test_default_no_checks_ready_returns_200(self) -> None:
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(create_health_blueprint())
        with app.test_client() as client:
            resp = client.get("/health/ready")
        assert resp.status_code == 200
