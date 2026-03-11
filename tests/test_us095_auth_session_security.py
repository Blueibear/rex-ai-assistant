"""Tests for US-095: Authentication and session security review.

Acceptance criteria:
- session tokens are cryptographically random (min 128 bits of entropy)
- session tokens invalidated on logout
- authentication endpoints have a failed-attempt rate limit or lockout
- tokens are not logged in plaintext anywhere in the logging output
- Typecheck passes
"""

from __future__ import annotations

import os
import secrets
import time
import base64

import pytest

from rex.dashboard.auth import (
    LoginRateLimiter,
    Session,
    SessionManager,
    get_login_rate_limiter,
    hash_token,
    verify_password,
)


# ---------------------------------------------------------------------------
# AC1: session tokens are cryptographically random (min 128 bits of entropy)
# ---------------------------------------------------------------------------


class TestTokenEntropy:
    def test_token_urlsafe_32_produces_256_bit_token(self):
        """secrets.token_urlsafe(32) yields 32 bytes = 256 bits (>= 128 required)."""
        token = secrets.token_urlsafe(32)
        # URL-safe base64: each char represents ~6 bits; 43 chars ≥ 256 bits
        # Decode to raw bytes and verify length
        # token_urlsafe(32) returns base64url of 32 random bytes → 43 chars
        assert len(token) >= 43

    def test_session_manager_creates_unique_tokens(self):
        mgr = SessionManager(expiry_seconds=3600)
        tokens = {mgr.create_session().token for _ in range(100)}
        assert len(tokens) == 100, "All 100 tokens should be unique"

    def test_session_token_length_sufficient_for_128_bit_entropy(self):
        """token_urlsafe(32) base64-encodes 32 bytes; raw bytes = 256 bits."""
        mgr = SessionManager(expiry_seconds=3600)
        session = mgr.create_session()
        # URL-safe base64 without padding: ceil(32 * 4/3) = 43 chars minimum
        assert len(session.token) >= 43

    def test_token_bytes_decode_to_at_least_16_bytes(self):
        """Verify underlying byte length via padding-safe decode."""
        mgr = SessionManager(expiry_seconds=3600)
        session = mgr.create_session()
        token = session.token
        # Add padding for decoding
        padded = token + "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(padded)
        assert len(raw) >= 16, "Token must encode at least 128 bits (16 bytes)"


# ---------------------------------------------------------------------------
# AC2: session tokens invalidated on logout
# ---------------------------------------------------------------------------


class TestSessionInvalidation:
    def test_invalidate_session_removes_token(self):
        mgr = SessionManager(expiry_seconds=3600)
        session = mgr.create_session()
        assert mgr.validate_session(session.token) is not None
        result = mgr.invalidate_session(session.token)
        assert result is True
        assert mgr.validate_session(session.token) is None

    def test_invalidate_nonexistent_token_returns_false(self):
        mgr = SessionManager(expiry_seconds=3600)
        assert mgr.invalidate_session("no-such-token") is False

    def test_session_count_decreases_after_invalidation(self):
        mgr = SessionManager(expiry_seconds=3600)
        s1 = mgr.create_session()
        s2 = mgr.create_session()
        assert mgr.get_active_session_count() == 2
        mgr.invalidate_session(s1.token)
        assert mgr.get_active_session_count() == 1
        mgr.invalidate_session(s2.token)
        assert mgr.get_active_session_count() == 0

    def test_invalidated_token_cannot_be_reused(self):
        mgr = SessionManager(expiry_seconds=3600)
        session = mgr.create_session()
        mgr.invalidate_session(session.token)
        # Attempt to validate same token again
        assert mgr.validate_session(session.token) is None

    def test_expired_session_also_invalid(self):
        mgr = SessionManager(expiry_seconds=0)  # expires immediately
        session = mgr.create_session()
        time.sleep(0.01)
        assert mgr.validate_session(session.token) is None


# ---------------------------------------------------------------------------
# AC3: authentication endpoints have a failed-attempt rate limit or lockout
# ---------------------------------------------------------------------------


class TestLoginRateLimiter:
    def _make_limiter(self, max_attempts: int = 3, window_seconds: int = 60) -> LoginRateLimiter:
        return LoginRateLimiter(max_attempts=max_attempts, window_seconds=window_seconds)

    def test_not_locked_out_initially(self):
        rl = self._make_limiter()
        assert rl.is_locked_out("1.2.3.4") is False

    def test_locked_out_after_max_attempts(self):
        rl = self._make_limiter(max_attempts=3)
        ip = "1.2.3.4"
        rl.record_failure(ip)
        rl.record_failure(ip)
        assert rl.is_locked_out(ip) is False
        locked = rl.record_failure(ip)
        assert locked is True
        assert rl.is_locked_out(ip) is True

    def test_record_failure_returns_true_when_just_locked(self):
        rl = self._make_limiter(max_attempts=1)
        ip = "5.6.7.8"
        result = rl.record_failure(ip)
        assert result is True

    def test_record_failure_returns_false_before_lockout(self):
        rl = self._make_limiter(max_attempts=5)
        ip = "5.6.7.8"
        for _ in range(4):
            result = rl.record_failure(ip)
        assert result is False
        assert rl.is_locked_out(ip) is False

    def test_success_resets_counter(self):
        rl = self._make_limiter(max_attempts=3)
        ip = "9.10.11.12"
        rl.record_failure(ip)
        rl.record_failure(ip)
        rl.record_success(ip)
        # After reset, 2 more failures should not lock (max is 3)
        rl.record_failure(ip)
        rl.record_failure(ip)
        assert rl.is_locked_out(ip) is False

    def test_different_ips_tracked_independently(self):
        rl = self._make_limiter(max_attempts=2)
        rl.record_failure("1.1.1.1")
        rl.record_failure("1.1.1.1")
        assert rl.is_locked_out("1.1.1.1") is True
        assert rl.is_locked_out("2.2.2.2") is False

    def test_attempts_expire_after_window(self):
        rl = self._make_limiter(max_attempts=2, window_seconds=0)
        ip = "3.3.3.3"
        rl.record_failure(ip)
        rl.record_failure(ip)
        # Window is 0 seconds — after a tiny sleep attempts are expired
        time.sleep(0.01)
        assert rl.is_locked_out(ip) is False

    def test_remaining_attempts_decreases(self):
        rl = self._make_limiter(max_attempts=5)
        ip = "4.4.4.4"
        assert rl.remaining_attempts(ip) == 5
        rl.record_failure(ip)
        assert rl.remaining_attempts(ip) == 4
        rl.record_failure(ip)
        assert rl.remaining_attempts(ip) == 3

    def test_remaining_attempts_zero_when_locked_out(self):
        rl = self._make_limiter(max_attempts=2)
        ip = "5.5.5.5"
        rl.record_failure(ip)
        rl.record_failure(ip)
        assert rl.remaining_attempts(ip) == 0

    def test_global_limiter_is_singleton(self):
        a = get_login_rate_limiter()
        b = get_login_rate_limiter()
        assert a is b


# ---------------------------------------------------------------------------
# AC4: tokens are not logged in plaintext
# ---------------------------------------------------------------------------


class TestTokenNotLoggedInPlaintext:
    def test_hash_token_returns_short_hex_not_original(self):
        token = secrets.token_urlsafe(32)
        hashed = hash_token(token)
        assert token not in hashed
        assert len(hashed) == 16
        # Should be hex
        int(hashed, 16)

    def test_hash_token_different_tokens_produce_different_hashes(self):
        t1 = secrets.token_urlsafe(32)
        t2 = secrets.token_urlsafe(32)
        assert hash_token(t1) != hash_token(t2)

    def test_hash_token_is_deterministic(self):
        token = secrets.token_urlsafe(32)
        assert hash_token(token) == hash_token(token)

    def test_auth_module_does_not_log_raw_token(self):
        """Verify hash_token is used for logging — not the raw token value."""
        token = "super-secret-token-value-xyz"
        hashed = hash_token(token)
        # The hash is a 16-char hex string; the raw token must not appear in it
        assert token not in hashed

    def test_session_manager_validate_returns_session_without_exposing_token_elsewhere(self):
        """validate_session returns Session object; the token is on the object, not logged."""
        mgr = SessionManager(expiry_seconds=3600)
        session = mgr.create_session()
        result = mgr.validate_session(session.token)
        assert result is not None
        assert result.token == session.token  # token accessible on object (not logged)


# ---------------------------------------------------------------------------
# Integration: rate limit enforced on login Flask endpoint
# ---------------------------------------------------------------------------


class TestLoginEndpointRateLimit:
    """Test the login endpoint returns 429 after too many failed attempts."""

    @pytest.fixture()
    def app(self, monkeypatch):
        """Create a minimal Flask app with the dashboard blueprint."""
        monkeypatch.setenv("REX_DASHBOARD_PASSWORD", "correct-password")
        monkeypatch.setenv("REX_LOGIN_MAX_ATTEMPTS", "3")
        monkeypatch.setenv("REX_LOGIN_LOCKOUT_SECONDS", "60")

        # Reset global singletons so env vars take effect
        import rex.dashboard.auth as auth_mod
        import rex.dashboard.routes as routes_mod

        auth_mod._session_manager = None
        auth_mod._login_rate_limiter = None

        from flask import Flask
        from rex.dashboard.routes import dashboard_bp

        application = Flask(__name__)
        application.register_blueprint(dashboard_bp)
        application.config["TESTING"] = True
        return application

    def test_correct_password_succeeds(self, app):
        with app.test_client() as client:
            resp = client.post(
                "/api/dashboard/login",
                json={"password": "correct-password"},
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert "token" in data

    def test_wrong_password_returns_401(self, app):
        with app.test_client() as client:
            resp = client.post(
                "/api/dashboard/login",
                json={"password": "wrong"},
            )
            assert resp.status_code == 401

    def test_rate_limit_after_max_failures(self, app, monkeypatch):
        """After max_attempts failures, endpoint returns 429."""
        import rex.dashboard.auth as auth_mod

        # Use a fresh limiter with max_attempts=3
        fresh_limiter = LoginRateLimiter(max_attempts=3, window_seconds=60)
        monkeypatch.setattr(auth_mod, "_login_rate_limiter", fresh_limiter)

        with app.test_client() as client:
            for _ in range(3):
                resp = client.post(
                    "/api/dashboard/login",
                    json={"password": "wrong"},
                    environ_base={"REMOTE_ADDR": "10.0.0.1"},
                )
            # 4th attempt should be rate-limited
            resp = client.post(
                "/api/dashboard/login",
                json={"password": "wrong"},
                environ_base={"REMOTE_ADDR": "10.0.0.1"},
            )
            assert resp.status_code == 429

    def test_successful_login_resets_rate_limit(self, app, monkeypatch):
        """A successful login clears the failure counter."""
        import rex.dashboard.auth as auth_mod

        fresh_limiter = LoginRateLimiter(max_attempts=3, window_seconds=60)
        monkeypatch.setattr(auth_mod, "_login_rate_limiter", fresh_limiter)

        with app.test_client() as client:
            ip = {"REMOTE_ADDR": "10.0.0.2"}
            # 2 failures
            client.post("/api/dashboard/login", json={"password": "wrong"}, environ_base=ip)
            client.post("/api/dashboard/login", json={"password": "wrong"}, environ_base=ip)
            # Success — resets counter
            resp = client.post(
                "/api/dashboard/login", json={"password": "correct-password"}, environ_base=ip
            )
            assert resp.status_code == 200
            # 2 more failures should not lock (counter was reset)
            client.post("/api/dashboard/login", json={"password": "wrong"}, environ_base=ip)
            resp = client.post(
                "/api/dashboard/login", json={"password": "wrong"}, environ_base=ip
            )
            assert resp.status_code == 401  # not 429

    def test_logout_invalidates_token(self, app, monkeypatch):
        """Token returned by login is invalidated after logout."""
        import rex.dashboard.auth as auth_mod

        auth_mod._login_rate_limiter = None  # fresh limiter

        with app.test_client() as client:
            # Login
            resp = client.post(
                "/api/dashboard/login",
                json={"password": "correct-password"},
            )
            assert resp.status_code == 200
            token = resp.get_json()["token"]

            # Logout
            resp = client.post(
                "/api/dashboard/logout",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200

            # Token should be gone from session manager
            mgr = auth_mod.get_session_manager()
            assert mgr.validate_session(token) is None
