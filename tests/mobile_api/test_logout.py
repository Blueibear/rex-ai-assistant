"""Logout, logout-all, and two-user isolation tests.

Matrix rows: REF-011..REF-015.
"""

from __future__ import annotations

from tests.mobile_api.conftest import auth_header, create_user, login_tokens


class TestLogout:
    def test_logout_revokes_access_and_refresh(self, client) -> None:
        create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")
        access = tokens["access_token"]

        assert client.post("/mobile/auth/logout", headers=auth_header(access)).status_code == 200
        # The otherwise-unexpired access token is invalidated by session state.
        assert client.get("/mobile/auth/session", headers=auth_header(access)).status_code == 401
        # And the refresh token is dead too.
        refresh = client.post(
            "/mobile/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
        )
        assert refresh.status_code == 401

    def test_repeated_logout_is_idempotent(self, client) -> None:
        """REF-012: repeated logout returns the same harmless success."""
        create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")
        access = tokens["access_token"]
        first = client.post("/mobile/auth/logout", headers=auth_header(access))
        second = client.post("/mobile/auth/logout", headers=auth_header(access))
        third = client.post("/mobile/auth/logout", headers=auth_header(access))
        for response in (first, second, third):
            assert response.status_code == 200
            assert response.get_json() == {"ok": True}

    def test_repeated_logout_does_not_modify_the_session(self, client, services) -> None:
        """The logout-only path never reactivates or mutates the session."""
        create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")
        access = tokens["access_token"]
        client.post("/mobile/auth/logout", headers=auth_header(access))
        before = dict(services.session_store.get_session(tokens["session_id"]))
        client.post("/mobile/auth/logout", headers=auth_header(access))
        after = dict(services.session_store.get_session(tokens["session_id"]))
        assert after == before
        assert after["revoked_at"] is not None
        assert after["revoke_reason"] == "logout"

    def test_logout_still_requires_full_token_validation(self, client) -> None:
        """The idempotent path never weakens JWT validation."""
        import jwt as pyjwt

        from tests.mobile_api.conftest import TEST_JWT_SECRET

        create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")
        claims = pyjwt.decode(
            tokens["access_token"],
            TEST_JWT_SECRET,
            algorithms=["HS256"],
            audience="askrex-mobile",
            issuer="askrex-assistant",
        )
        # Wrong signature is still rejected on logout.
        forged = pyjwt.encode(claims, "w" * 64, algorithm="HS256")
        assert client.post("/mobile/auth/logout", headers=auth_header(forged)).status_code == 401
        # A session owned by someone else is still rejected on logout.
        other_id = create_user("sarah", "pw-abcdef")
        claims["sub"] = other_id
        crossed = pyjwt.encode(claims, TEST_JWT_SECRET, algorithm="HS256")
        assert client.post("/mobile/auth/logout", headers=auth_header(crossed)).status_code == 401
        # Missing auth is still rejected.
        assert client.post("/mobile/auth/logout").status_code == 401

    def test_revoked_token_fails_on_every_other_protected_endpoint(self, client) -> None:
        """Revoked sessions are permitted on /logout only — nowhere else."""
        create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")
        access = tokens["access_token"]
        assert client.post("/mobile/auth/logout", headers=auth_header(access)).status_code == 200

        protected = [
            ("get", "/mobile/auth/session"),
            ("post", "/mobile/auth/logout-all"),
            ("post", "/mobile/chat"),
            ("post", "/mobile/voice/upload"),
            ("get", "/mobile/notifications"),
            ("get", "/mobile/settings"),
        ]
        for method, path in protected:
            response = getattr(client, method)(path, headers=auth_header(access))
            assert response.status_code == 401, path
            assert response.get_json()["error"]["code"] == "AUTH_SESSION_REVOKED", path


class TestLogoutAll:
    def test_logout_all_revokes_every_own_session(self, client) -> None:
        create_user("james", "pw-123456")
        first = login_tokens(client, "james", "pw-123456")
        second = login_tokens(client, "james", "pw-123456")

        response = client.post(
            "/mobile/auth/logout-all", headers=auth_header(second["access_token"])
        )
        assert response.status_code == 200
        assert response.get_json()["revoked_sessions"] == 2

        for tokens in (first, second):
            assert (
                client.get(
                    "/mobile/auth/session",
                    headers=auth_header(tokens["access_token"]),
                ).status_code
                == 401
            )

    def test_logout_all_does_not_touch_other_users(self, client) -> None:
        """REF-014: logout-all is scoped to the authenticated user only."""
        create_user("james", "pw-123456")
        create_user("sarah", "pw-abcdef")
        james = login_tokens(client, "james", "pw-123456")
        sarah = login_tokens(client, "sarah", "pw-abcdef")

        client.post("/mobile/auth/logout-all", headers=auth_header(james["access_token"]))

        assert (
            client.get(
                "/mobile/auth/session", headers=auth_header(james["access_token"])
            ).status_code
            == 401
        )
        # Sarah's session is untouched.
        still_active = client.get(
            "/mobile/auth/session", headers=auth_header(sarah["access_token"])
        )
        assert still_active.status_code == 200

    def test_users_refresh_tokens_stay_isolated(self, client) -> None:
        """REF-015: one user's refresh token cannot act on another's session."""
        create_user("james", "pw-123456")
        create_user("sarah", "pw-abcdef")
        james = login_tokens(client, "james", "pw-123456")
        sarah = login_tokens(client, "sarah", "pw-abcdef")

        client.post("/mobile/auth/logout-all", headers=auth_header(james["access_token"]))
        # James's refresh is revoked; Sarah's still rotates.
        assert (
            client.post(
                "/mobile/auth/refresh",
                json={"refresh_token": james["refresh_token"]},
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/mobile/auth/refresh",
                json={"refresh_token": sarah["refresh_token"]},
            ).status_code
            == 200
        )
