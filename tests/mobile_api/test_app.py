"""Application factory and middleware foundation tests.

Matrix rows: FND-001, FND-002, FND-009..FND-015, plus default rate limiting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.mobile_api.conftest import create_user


class TestImportSideEffects:
    def test_import_does_not_touch_database(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FND-001: importing the package must not create or mutate the DB."""
        import importlib

        data_dir = tmp_path / "data"
        monkeypatch.setenv("REX_DATA_DIR", str(data_dir))

        import rex.mobile_api
        import rex.mobile_api.app

        importlib.reload(rex.mobile_api)
        assert not data_dir.exists()


class TestAppFactory:
    def test_two_apps_are_independent(self, services) -> None:
        """FND-002: the factory returns independent app instances."""
        from rex.mobile_api.app import create_mobile_app

        app_one = create_mobile_app(services=services)
        app_two = create_mobile_app(services=services)
        assert app_one is not app_two
        assert (
            app_one.extensions["mobile_api_limiter"] is not app_two.extensions["mobile_api_limiter"]
        )

    def test_unknown_route_returns_nested_404(self, client) -> None:
        """FND-009: unknown routes use the canonical envelope with request ID."""
        response = client.get("/mobile/does-not-exist")
        assert response.status_code == 404
        body = response.get_json()
        assert body["error"]["code"] == "NOT_FOUND"
        assert body["error"]["retryable"] is False
        assert body["error"]["request_id"]

    def test_unexpected_exception_returns_generic_500(self, app) -> None:
        """FND-010: unexpected exceptions leak no stack, path, or secret."""
        state = {"secret_path": "C:/very/secret/model/path"}

        @app.route("/mobile/_boom")
        def _boom():  # pragma: no cover - exercised via test client
            raise RuntimeError(f"exploded near {state['secret_path']}")

        app.config["TESTING"] = False
        app.config["PROPAGATE_EXCEPTIONS"] = False
        with app.test_client() as client:
            response = client.get("/mobile/_boom")
        assert response.status_code == 500
        body = response.get_json()
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert "secret" not in body["error"]["message"]
        assert state["secret_path"] not in response.get_data(as_text=True)

    def test_request_id_header_matches_error_field(self, client) -> None:
        """FND-011: the header and error field carry the same request ID."""
        response = client.get("/mobile/does-not-exist")
        assert response.headers["X-Request-ID"] == response.get_json()["error"]["request_id"]

    def test_api_version_header_on_every_response(self, client) -> None:
        """FND-012: X-AskRex-API-Version is present on success and error."""
        ok = client.get("/mobile/status")
        missing = client.get("/mobile/does-not-exist")
        assert ok.headers["X-AskRex-API-Version"] == "1.0"
        assert missing.headers["X-AskRex-API-Version"] == "1.0"

    def test_unsupported_content_type_rejected(self, client) -> None:
        """FND-013: non-JSON content type is rejected before business logic."""
        response = client.post(
            "/mobile/auth/login",
            data="username=james&password=x",
            content_type="application/x-www-form-urlencoded",
        )
        assert response.status_code == 415
        assert response.get_json()["error"]["code"] == "INVALID_MEDIA"

    def test_oversized_json_body_rejected(self, mobile_env: Path, clock) -> None:
        """FND-014: bodies over max_json_bytes get 413 before processing."""
        from rex.config import MobileApiConfig
        from rex.mobile_api.app import create_mobile_app
        from rex.mobile_api.db import migrate_users_db
        from rex.mobile_api.services import MobileApiServices

        db_path = mobile_env / "users.db"
        migrate_users_db(db_path)
        config = MobileApiConfig(max_json_bytes=256)
        services = MobileApiServices.build(config, db_path=db_path, clock=clock)
        app = create_mobile_app(services=services)
        app.config["TESTING"] = True
        with app.test_client() as client:
            response = client.post(
                "/mobile/auth/login",
                json={"username": "a", "password": "b" * 1024},
            )
        assert response.status_code == 413
        assert response.get_json()["error"]["code"] == "PAYLOAD_TOO_LARGE"

    def test_cors_denied_by_default(self, client) -> None:
        """FND-015: no Access-Control-Allow-Origin unless explicitly configured."""
        response = client.get("/mobile/status", headers={"Origin": "https://evil.example"})
        assert "Access-Control-Allow-Origin" not in response.headers

    def test_cors_allows_only_configured_origin(self, mobile_env: Path, clock) -> None:
        from rex.config import MobileApiConfig
        from rex.mobile_api.app import create_mobile_app
        from rex.mobile_api.db import migrate_users_db
        from rex.mobile_api.services import MobileApiServices

        db_path = mobile_env / "users.db"
        migrate_users_db(db_path)
        config = MobileApiConfig(allowed_origins=["https://app.askrex.local"])
        services = MobileApiServices.build(config, db_path=db_path, clock=clock)
        app = create_mobile_app(services=services)
        app.config["TESTING"] = True
        with app.test_client() as client:
            allowed = client.get("/mobile/status", headers={"Origin": "https://app.askrex.local"})
            denied = client.get("/mobile/status", headers={"Origin": "https://evil.example"})
        assert allowed.headers.get("Access-Control-Allow-Origin") == "https://app.askrex.local"
        assert "Access-Control-Allow-Origin" not in denied.headers


class TestDefaultRateLimit:
    def test_default_limit_returns_canonical_429(self, mobile_env: Path, clock) -> None:
        from rex.config import MobileApiConfig
        from rex.mobile_api.app import create_mobile_app
        from rex.mobile_api.db import migrate_users_db
        from rex.mobile_api.services import MobileApiServices

        db_path = mobile_env / "users.db"
        migrate_users_db(db_path)
        config = MobileApiConfig(rate_limit_default="2 per minute")
        services = MobileApiServices.build(config, db_path=db_path, clock=clock)
        app = create_mobile_app(services=services)
        app.config["TESTING"] = True
        with app.test_client() as client:
            assert client.get("/mobile/status").status_code == 200
            assert client.get("/mobile/status").status_code == 200
            limited = client.get("/mobile/status")
        assert limited.status_code == 429
        body = limited.get_json()
        assert body["error"]["code"] == "RATE_LIMITED"
        assert body["error"]["retryable"] is True
        assert int(limited.headers["Retry-After"]) >= 1


class TestClientIdentityFieldsIgnored:
    def test_login_ignores_client_role_and_permissions(self, client) -> None:
        """AUTH-021: client-supplied role/permissions never become authority."""
        create_user("james", "pw-123456", admin=False)
        response = client.post(
            "/mobile/auth/login",
            json={
                "username": "james",
                "password": "pw-123456",
                "role": "owner",
                "permissions": ["admin"],
                "risk_level": "none",
                "approved": True,
            },
        )
        assert response.status_code == 200
        user = response.get_json()["user"]
        assert user["role"] == "member"
        assert user["permissions"] == []
