"""Login endpoint tests.

Matrix rows: AUTH-001..AUTH-007, AUTH-024, USR-005, USR-011.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.mobile_api.conftest import (
    TEST_JWT_SECRET,
    create_user,
    disable_user,
    login,
)


def _sessions(db_path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM mobile_sessions").fetchall()
    finally:
        conn.close()


class TestLoginSuccess:
    def test_valid_login_returns_token_pair_and_projection(self, client, services) -> None:
        user_id = create_user("james", "pw-123456", admin=True)
        response = login(client, "james", "pw-123456")
        assert response.status_code == 200
        body = response.get_json()
        assert body["token_type"] == "Bearer"
        assert body["expires_in"] == 900
        assert body["refresh_expires_in"] == 30 * 86400
        assert body["access_token"]
        assert body["refresh_token"]
        assert body["session_id"]
        assert body["user"]["id"] == user_id
        assert body["user"]["role"] == "owner"
        assert body["user"]["permissions"] == ["admin"]
        assert body["user"]["name"] == "james"

    def test_device_metadata_stored_with_session(self, client, services) -> None:
        create_user("james", "pw-123456")
        response = login(
            client,
            "james",
            "pw-123456",
            device={
                "device_id": "stable-random-device-01",
                "name": "James's iPhone",
                "platform": "ios",
                "app_version": "0.1.0",
            },
        )
        assert response.status_code == 200
        rows = _sessions(services.db_path)
        assert len(rows) == 1
        assert rows[0]["device_id"] == "stable-random-device-01"
        assert rows[0]["platform"] == "ios"

    def test_missing_device_gets_generated_id(self, client, services) -> None:
        create_user("james", "pw-123456")
        assert login(client, "james", "pw-123456").status_code == 200
        rows = _sessions(services.db_path)
        assert len(rows) == 1
        assert rows[0]["device_id"]


class TestLoginFailure:
    def test_invalid_password_is_401_without_enumeration(self, client) -> None:
        create_user("james", "pw-123456")
        wrong_password = login(client, "james", "wrong-password")
        unknown_user = login(client, "no-such-user", "wrong-password")
        assert wrong_password.status_code == 401
        assert unknown_user.status_code == 401
        # USR-011: externally indistinguishable errors.
        assert wrong_password.get_json()["error"]["code"] == "AUTH_INVALID_CREDENTIALS"
        assert (
            wrong_password.get_json()["error"]["message"]
            == unknown_user.get_json()["error"]["message"]
        )

    def test_disabled_user_cannot_login(self, client, services) -> None:
        user_id = create_user("james", "pw-123456")
        disable_user(services.db_path, user_id)
        response = login(client, "james", "pw-123456")
        assert response.status_code == 401
        assert response.get_json()["error"]["code"] == "AUTH_INVALID_CREDENTIALS"

    def test_missing_fields_return_400(self, client) -> None:
        assert client.post("/mobile/auth/login", json={}).status_code == 400
        assert client.post("/mobile/auth/login", json={"username": "james"}).status_code == 400
        assert client.post("/mobile/auth/login", json={"password": "x"}).status_code == 400

    def test_malformed_json_returns_400(self, client) -> None:
        response = client.post(
            "/mobile/auth/login",
            data="{not-json",
            content_type="application/json",
        )
        assert response.status_code == 400
        assert response.get_json()["error"]["code"] == "BAD_REQUEST"

    def test_non_object_json_returns_400(self, client) -> None:
        response = client.post(
            "/mobile/auth/login", data='"a string"', content_type="application/json"
        )
        assert response.status_code == 400

    def test_invalid_device_id_rejected(self, client) -> None:
        create_user("james", "pw-123456")
        response = login(
            client,
            "james",
            "pw-123456",
            device={"device_id": "../../etc/passwd"},
        )
        assert response.status_code == 400

    def test_overlong_device_id_rejected(self, client) -> None:
        create_user("james", "pw-123456")
        response = login(client, "james", "pw-123456", device={"device_id": "a" * 200})
        assert response.status_code == 400


class TestLoginRateLimit:
    def test_login_rate_limit_applies(self, mobile_env: Path, clock) -> None:
        """AUTH-024: repeated logins hit the login-specific limit with 429."""
        from rex.config import MobileApiConfig
        from rex.mobile_api.app import create_mobile_app
        from rex.mobile_api.db import migrate_users_db
        from rex.mobile_api.services import MobileApiServices

        db_path = mobile_env / "users.db"
        migrate_users_db(db_path)
        config = MobileApiConfig(
            rate_limit_login="2 per minute", rate_limit_default="100 per minute"
        )
        services = MobileApiServices.build(config, db_path=db_path, clock=clock)
        app = create_mobile_app(services=services)
        app.config["TESTING"] = True
        with app.test_client() as client:
            for _ in range(2):
                login(client, "james", "bad-password")
            limited = login(client, "james", "bad-password")
        assert limited.status_code == 429
        assert limited.get_json()["error"]["code"] == "RATE_LIMITED"
        assert "Retry-After" in limited.headers


class TestLoginRateKey:
    """The login limiter key must never contain raw IPs or usernames."""

    def _key(self, app, username, address: str = "203.0.113.7") -> str:
        from rex.mobile_api.routes.auth import _login_rate_key

        with app.test_request_context(
            "/mobile/auth/login",
            method="POST",
            json={"username": username, "password": "x"},
            environ_base={"REMOTE_ADDR": address},
        ):
            return _login_rate_key()

    def test_raw_ip_and_username_absent_from_key(self, app) -> None:
        key = self._key(app, "james", address="203.0.113.7")
        assert "203.0.113.7" not in key
        assert "james" not in key

    def test_username_normalization_is_consistent(self, app) -> None:
        assert self._key(app, "James ") == self._key(app, "james")
        assert self._key(app, "  JAMES") == self._key(app, "james")

    def test_distinct_usernames_and_addresses_produce_distinct_keys(self, app) -> None:
        base = self._key(app, "james", address="203.0.113.7")
        assert self._key(app, "sarah", address="203.0.113.7") != base
        assert self._key(app, "james", address="203.0.113.99") != base


class TestLoginSecrecy:
    def test_password_and_tokens_never_logged(self, client, caplog) -> None:
        """AUTH-007: raw credentials and tokens are absent from captured logs."""
        import logging

        create_user("james", "super-secret-pw")
        with caplog.at_level(logging.DEBUG):
            response = login(client, "james", "super-secret-pw")
        assert response.status_code == 200
        body = response.get_json()
        log_text = " ".join(record.getMessage() for record in caplog.records)
        assert "super-secret-pw" not in log_text
        assert body["access_token"] not in log_text
        assert body["refresh_token"] not in log_text
        assert TEST_JWT_SECRET not in log_text
