"""Access JWT validation tests using crafted tokens.

Matrix rows: AUTH-008..AUTH-020, SES-002..SES-004, USR-006, USR-012.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt as pyjwt

from tests.mobile_api.conftest import (
    TEST_JWT_SECRET,
    auth_header,
    create_user,
    disable_user,
    login_tokens,
)

_SESSION_URL = "/mobile/auth/session"


def _claims(
    *,
    sub: str,
    sid: str,
    iss: str = "askrex-assistant",
    aud: str = "askrex-mobile",
    ttl_seconds: int = 900,
    nbf_offset: int = 0,
) -> dict:
    now = datetime.now(UTC)
    return {
        "iss": iss,
        "aud": aud,
        "sub": sub,
        "sid": sid,
        "jti": "crafted-token-id",
        "iat": now,
        "nbf": now + timedelta(seconds=nbf_offset),
        "exp": now + timedelta(seconds=ttl_seconds),
    }


def _encode(claims: dict, secret: str = TEST_JWT_SECRET, algorithm: str = "HS256") -> str:
    return pyjwt.encode(claims, secret, algorithm=algorithm)


class TestAccessTokenClaims:
    def test_login_token_contains_required_claims(self, client) -> None:
        """AUTH-008: iss/aud/sub/sid/jti/iat/nbf/exp are all present."""
        user_id = create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")
        claims = pyjwt.decode(
            tokens["access_token"],
            TEST_JWT_SECRET,
            algorithms=["HS256"],
            audience="askrex-mobile",
            issuer="askrex-assistant",
        )
        for name in ("iss", "aud", "sub", "sid", "jti", "iat", "nbf", "exp"):
            assert name in claims
        assert claims["sub"] == user_id
        assert claims["sid"] == tokens["session_id"]
        assert claims["exp"] - claims["iat"] == 900


class TestTokenRejection:
    def _login(self, client) -> tuple[str, str]:
        user_id = create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")
        return user_id, tokens["session_id"]

    def _assert_401(self, client, token: str, code: str) -> None:
        response = client.get(_SESSION_URL, headers=auth_header(token))
        assert response.status_code == 401
        assert response.get_json()["error"]["code"] == code

    def test_wrong_signature_rejected(self, client) -> None:
        user_id, sid = self._login(client)
        token = _encode(_claims(sub=user_id, sid=sid), secret="w" * 64)
        self._assert_401(client, token, "AUTH_TOKEN_INVALID")

    def test_wrong_algorithm_rejected(self, client) -> None:
        user_id, sid = self._login(client)
        token = _encode(_claims(sub=user_id, sid=sid), algorithm="HS512")
        self._assert_401(client, token, "AUTH_TOKEN_INVALID")

    def test_wrong_issuer_rejected(self, client) -> None:
        user_id, sid = self._login(client)
        token = _encode(_claims(sub=user_id, sid=sid, iss="evil-issuer"))
        self._assert_401(client, token, "AUTH_TOKEN_INVALID")

    def test_wrong_audience_rejected(self, client) -> None:
        user_id, sid = self._login(client)
        token = _encode(_claims(sub=user_id, sid=sid, aud="askrex-desktop"))
        self._assert_401(client, token, "AUTH_TOKEN_INVALID")

    def test_expired_token_rejected(self, client) -> None:
        user_id, sid = self._login(client)
        token = _encode(_claims(sub=user_id, sid=sid, ttl_seconds=-120))
        self._assert_401(client, token, "AUTH_TOKEN_EXPIRED")

    def test_future_nbf_rejected(self, client) -> None:
        user_id, sid = self._login(client)
        token = _encode(_claims(sub=user_id, sid=sid, nbf_offset=600))
        self._assert_401(client, token, "AUTH_TOKEN_INVALID")

    def test_missing_required_claim_rejected(self, client) -> None:
        user_id, sid = self._login(client)
        claims = _claims(sub=user_id, sid=sid)
        del claims["sid"]
        self._assert_401(client, _encode(claims), "AUTH_TOKEN_INVALID")

    def test_invalid_sub_fails_before_private_access(self, client, monkeypatch) -> None:
        """AUTH-016 / USR-012: a non-canonical sub never reaches user lookup."""
        _, sid = self._login(client)

        reached = {"user_lookup": False}
        from rex.mobile_api import users as musers

        original = musers.get_user

        def _tracking_get_user(conn, user_id):
            reached["user_lookup"] = True
            return original(conn, user_id)

        monkeypatch.setattr(musers, "get_user", _tracking_get_user)
        for bad_sub in ("..", "../../etc", "con", "a/b"):
            token = _encode(_claims(sub=bad_sub, sid=sid))
            self._assert_401(client, token, "AUTH_TOKEN_INVALID")
        assert reached["user_lookup"] is False

    def test_unknown_session_rejected(self, client) -> None:
        user_id, _ = self._login(client)
        token = _encode(_claims(sub=user_id, sid="00000000-0000-0000-0000-000000000000"))
        self._assert_401(client, token, "AUTH_TOKEN_INVALID")

    def test_session_owned_by_other_user_rejected(self, client) -> None:
        """AUTH-018 / SES-008: cross-user session use is indistinguishable."""
        create_user("james", "pw-123456")
        tokens_a = login_tokens(client, "james", "pw-123456")
        other_id = create_user("mallory", "pw-abcdef")
        token = _encode(_claims(sub=other_id, sid=tokens_a["session_id"]))
        # Same code/message as an unknown session: no session enumeration.
        self._assert_401(client, token, "AUTH_TOKEN_INVALID")

    def test_revoked_session_with_unexpired_jwt_rejected(self, client) -> None:
        """AUTH-019: logout invalidates otherwise-unexpired access tokens."""
        create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")
        access = tokens["access_token"]
        assert client.post("/mobile/auth/logout", headers=auth_header(access)).status_code == 200
        self._assert_401(client, access, "AUTH_SESSION_REVOKED")

    def test_expired_session_with_unexpired_jwt_rejected(self, client, clock) -> None:
        """AUTH-020: session expiry wins even while the JWT is still valid."""
        create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")
        clock.advance(days=31)
        self._assert_401(client, tokens["access_token"], "AUTH_SESSION_REVOKED")

    def test_deleted_user_with_live_session_rejected(self, client, services) -> None:
        """USR-006: access fails closed when the user vanishes."""
        from rex.mobile_api.db import connect

        user_id = create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")
        conn = connect(services.db_path)
        try:
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        finally:
            conn.close()
        self._assert_401(client, tokens["access_token"], "AUTH_SESSION_REVOKED")

    def test_disabled_user_with_live_session_rejected(self, client, services) -> None:
        user_id = create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")
        disable_user(services.db_path, user_id)
        self._assert_401(client, tokens["access_token"], "AUTH_SESSION_REVOKED")


class TestAuthorizationHeaderSyntax:
    def test_missing_header_rejected(self, client) -> None:
        response = client.get(_SESSION_URL)
        assert response.status_code == 401

    def test_wrong_scheme_rejected(self, client) -> None:
        create_user("james", "pw-123456")
        response = client.get(_SESSION_URL, headers={"Authorization": "Basic amFtZXM6cHc="})
        assert response.status_code == 401

    def test_empty_bearer_rejected(self, client) -> None:
        response = client.get(_SESSION_URL, headers={"Authorization": "Bearer "})
        assert response.status_code == 401
