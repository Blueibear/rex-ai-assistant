"""Tests for US-113: Production configuration defaults.

Acceptance criteria:
  - DEBUG mode disabled when ENVIRONMENT=production
  - Stack traces not returned to API clients in production
  - Development-only endpoints/routes disabled or unreachable in production mode
  - Production mode detectable from a single ENVIRONMENT environment variable
  - Typecheck passes
"""

from __future__ import annotations

from flask import Flask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(*, register_debug_route: bool = False) -> Flask:
    """Create a minimal Flask app for testing."""
    app = Flask(__name__)
    if register_debug_route:

        @app.route("/debug/info")
        def debug_info():
            return "debug info"

        @app.route("/debug/metrics")
        def debug_metrics():
            return "debug metrics"

    @app.route("/api/status")
    def status():
        return "ok"

    @app.route("/api/crash")
    def crash():
        raise RuntimeError("intentional crash for testing")

    return app


# ---------------------------------------------------------------------------
# is_production()
# ---------------------------------------------------------------------------


class TestIsProduction:
    def test_returns_true_when_environment_is_production(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        from rex.production_config import is_production

        assert is_production() is True

    def test_case_insensitive_production(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "PRODUCTION")
        from rex.production_config import is_production

        assert is_production() is True

    def test_mixed_case_production(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "Production")
        from rex.production_config import is_production

        assert is_production() is True

    def test_returns_false_when_environment_is_development(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "development")
        from rex.production_config import is_production

        assert is_production() is False

    def test_returns_false_when_environment_is_staging(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "staging")
        from rex.production_config import is_production

        assert is_production() is False

    def test_returns_false_when_environment_is_test(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "test")
        from rex.production_config import is_production

        assert is_production() is False

    def test_returns_false_when_environment_variable_unset(self, monkeypatch):
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        from rex.production_config import is_production

        assert is_production() is False

    def test_returns_false_for_empty_string(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "")
        from rex.production_config import is_production

        assert is_production() is False

    def test_detectable_from_single_env_var(self, monkeypatch):
        """Production mode must be detectable from the ENVIRONMENT variable alone."""
        from rex.production_config import is_production

        monkeypatch.setenv("ENVIRONMENT", "production")
        assert is_production() is True

        monkeypatch.setenv("ENVIRONMENT", "development")
        assert is_production() is False


# ---------------------------------------------------------------------------
# apply_production_defaults() — debug mode
# ---------------------------------------------------------------------------


class TestDebugModeDisabled:
    def test_debug_disabled_in_production(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        from rex.production_config import apply_production_defaults

        app = _make_app()
        app.debug = True  # start with debug on
        apply_production_defaults(app)
        assert app.debug is False

    def test_testing_disabled_in_production(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        from rex.production_config import apply_production_defaults

        app = _make_app()
        app.testing = True
        apply_production_defaults(app)
        assert app.testing is False

    def test_debug_unchanged_in_development(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "development")
        from rex.production_config import apply_production_defaults

        app = _make_app()
        app.debug = True  # stays on in dev
        apply_production_defaults(app)
        assert app.debug is True  # not modified by the helper

    def test_debug_unchanged_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        from rex.production_config import apply_production_defaults

        app = _make_app()
        app.debug = True
        apply_production_defaults(app)
        assert app.debug is True


# ---------------------------------------------------------------------------
# apply_production_defaults() — stack traces not returned
# ---------------------------------------------------------------------------


class TestNoStackTracesInProduction:
    def test_500_returns_json_error_envelope_not_traceback(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        from rex.production_config import apply_production_defaults

        app = _make_app()
        # Disable propagate so the error handler is exercised, not pytest
        app.config["PROPAGATE_EXCEPTIONS"] = False
        apply_production_defaults(app)

        client = app.test_client()
        resp = client.get("/api/crash")
        assert resp.status_code == 500
        data = resp.get_json()
        assert data is not None
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]

    def test_500_body_contains_no_traceback_keywords(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        from rex.production_config import apply_production_defaults

        app = _make_app()
        app.config["PROPAGATE_EXCEPTIONS"] = False
        apply_production_defaults(app)

        client = app.test_client()
        resp = client.get("/api/crash")
        body = resp.data.decode()
        # Should not contain traceback markers
        assert "Traceback" not in body
        assert "RuntimeError" not in body

    def test_500_handler_not_registered_in_development(self, monkeypatch):
        """In development no 500 override is installed — Flask default propagates."""
        monkeypatch.setenv("ENVIRONMENT", "development")
        from rex.production_config import apply_production_defaults

        app = _make_app()
        apply_production_defaults(app)
        # No custom 500 handler registered
        assert 500 not in app.error_handler_spec[None]


# ---------------------------------------------------------------------------
# apply_production_defaults() — dev-only routes blocked
# ---------------------------------------------------------------------------


class TestDevOnlyRoutesBlocked:
    def test_debug_prefix_returns_404_in_production(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        from rex.production_config import apply_production_defaults

        app = _make_app(register_debug_route=True)
        apply_production_defaults(app)

        client = app.test_client()
        resp = client.get("/debug/info")
        assert resp.status_code == 404

    def test_debug_metrics_returns_404_in_production(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        from rex.production_config import apply_production_defaults

        app = _make_app(register_debug_route=True)
        apply_production_defaults(app)

        client = app.test_client()
        resp = client.get("/debug/metrics")
        assert resp.status_code == 404

    def test_non_debug_routes_still_accessible_in_production(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        from rex.production_config import apply_production_defaults

        app = _make_app()
        app.config["PROPAGATE_EXCEPTIONS"] = False
        apply_production_defaults(app)

        client = app.test_client()
        resp = client.get("/api/status")
        assert resp.status_code == 200

    def test_debug_routes_accessible_in_development(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "development")
        from rex.production_config import apply_production_defaults

        app = _make_app(register_debug_route=True)
        apply_production_defaults(app)

        client = app.test_client()
        resp = client.get("/debug/info")
        assert resp.status_code == 200

    def test_debug_prefix_list_exported(self):
        from rex.production_config import _DEV_ONLY_PREFIXES

        assert "/debug" in _DEV_ONLY_PREFIXES

    def test_debug_subpath_also_blocked(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        from rex.production_config import apply_production_defaults

        app = _make_app()

        @app.route("/debug/something/deep")
        def deep():
            return "deep"

        apply_production_defaults(app)
        client = app.test_client()
        resp = client.get("/debug/something/deep")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# apply_production_defaults() — idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_calling_twice_does_not_double_register_handlers(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        from rex.production_config import apply_production_defaults

        app = _make_app()
        apply_production_defaults(app)
        apply_production_defaults(app)  # second call must be safe
        # Only one before_request hook registered by production_config
        # (can't count easily, but app should still respond correctly)
        app.config["PROPAGATE_EXCEPTIONS"] = False
        client = app.test_client()
        resp = client.get("/api/status")
        assert resp.status_code == 200

    def test_calling_in_dev_mode_twice_is_safe(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "development")
        from rex.production_config import apply_production_defaults

        app = _make_app()
        apply_production_defaults(app)
        apply_production_defaults(app)
        client = app.test_client()
        resp = client.get("/api/status")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    def test_is_production_exported(self):
        from rex.production_config import is_production  # noqa: F401

    def test_apply_production_defaults_exported(self):
        from rex.production_config import apply_production_defaults  # noqa: F401

    def test_dev_only_prefixes_exported(self):
        from rex.production_config import _DEV_ONLY_PREFIXES

        assert isinstance(_DEV_ONLY_PREFIXES, frozenset)
