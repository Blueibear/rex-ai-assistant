"""Dashboard authentication utilities.

Provides session-based authentication for the dashboard API endpoints.
Tokens are stored in-memory with configurable expiration.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from rex.config_manager import load_config

# Session expiry in seconds (default: 8 hours)
SESSION_EXPIRY_SECONDS = int(os.getenv("REX_DASHBOARD_SESSION_EXPIRY") or "28800")

# Secret key for HMAC token signing - auto-generated if not set
_TOKEN_SECRET = os.getenv("REX_DASHBOARD_SECRET") or secrets.token_hex(32)


@dataclass
class Session:
    """Represents an authenticated dashboard session."""

    token: str
    created_at: datetime
    expires_at: datetime
    user_key: str = "dashboard"
    metadata: dict = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return datetime.now() >= self.expires_at


class SessionManager:
    """Thread-safe session manager for dashboard authentication."""

    def __init__(self, expiry_seconds: int = SESSION_EXPIRY_SECONDS) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()
        self._expiry_seconds = expiry_seconds

    def create_session(self, user_key: str = "dashboard", metadata: dict = None) -> Session:  # type: ignore[assignment]
        """Create a new session and return the session object."""
        token = secrets.token_urlsafe(32)
        now = datetime.now()
        expires_at = now + timedelta(seconds=self._expiry_seconds)

        session = Session(
            token=token,
            created_at=now,
            expires_at=expires_at,
            user_key=user_key,
            metadata=metadata or {},
        )

        with self._lock:
            # Clean up expired sessions periodically
            self._cleanup_expired()
            self._sessions[token] = session

        return session

    def validate_session(self, token: str) -> Session | None:
        """Validate a session token and return the session if valid."""
        if not token:
            return None

        with self._lock:
            session = self._sessions.get(token)
            if session is None:
                return None

            if session.is_expired():
                del self._sessions[token]
                return None

            return session

    def invalidate_session(self, token: str) -> bool:
        """Invalidate a session token. Returns True if session existed."""
        with self._lock:
            if token in self._sessions:
                del self._sessions[token]
                return True
            return False

    def _cleanup_expired(self) -> None:
        """Remove expired sessions (called internally)."""
        now = datetime.now()
        expired = [t for t, s in self._sessions.items() if s.expires_at <= now]
        for token in expired:
            del self._sessions[token]

    def get_active_session_count(self) -> int:
        """Get number of active sessions."""
        with self._lock:
            self._cleanup_expired()
            return len(self._sessions)


# Global session manager instance
_session_manager: SessionManager | None = None
_session_manager_lock = threading.Lock()


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        with _session_manager_lock:
            if _session_manager is None:
                _session_manager = SessionManager()
    return _session_manager


def get_dashboard_password() -> str | None:
    """Get the dashboard password from config or environment.

    Checks in order:
    1. REX_DASHBOARD_PASSWORD environment variable
    2. dashboard.password in rex_config.json

    Returns None if no password is configured (local-only access mode).
    """
    # Check environment first
    env_password = os.getenv("REX_DASHBOARD_PASSWORD")
    if env_password:
        return env_password

    # Check config file
    try:
        config = load_config()
        dashboard_config = config.get("dashboard", {})
        if isinstance(dashboard_config, dict):
            return dashboard_config.get("password")
    except Exception:
        pass

    return None


def verify_password(password: str) -> bool:
    """Verify the provided password against the configured password.

    Uses constant-time comparison to prevent timing attacks.
    """
    expected = get_dashboard_password()
    if expected is None:
        # No password configured - only allow if explicitly permitted
        return False

    # Use constant-time comparison
    return hmac.compare_digest(password.encode(), expected.encode())


def is_password_required() -> bool:
    """Check if password authentication is required.

    Returns True if a dashboard password is configured.
    """
    return get_dashboard_password() is not None


def hash_token(token: str) -> str:
    """Create a hash of a token for logging (non-reversible)."""
    return hashlib.sha256(f"{token}:{_TOKEN_SECRET}".encode()).hexdigest()[:16]


# Login rate-limiter defaults (overridable via env vars)
_LOGIN_MAX_ATTEMPTS = int(os.getenv("REX_LOGIN_MAX_ATTEMPTS") or "5")
_LOGIN_LOCKOUT_SECONDS = int(os.getenv("REX_LOGIN_LOCKOUT_SECONDS") or "300")


class LoginRateLimiter:
    """Per-IP failed-login attempt tracker with timed lockout.

    An IP is locked out once it accumulates ``max_attempts`` failures
    within ``window_seconds``.  A successful login resets the counter.
    """

    def __init__(
        self,
        max_attempts: int = _LOGIN_MAX_ATTEMPTS,
        window_seconds: int = _LOGIN_LOCKOUT_SECONDS,
    ) -> None:
        self._max_attempts = max_attempts
        self._window_seconds = window_seconds
        # {ip: [timestamp, ...]}
        self._attempts: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, ip: str, now: float) -> None:
        """Remove attempts older than the window (must hold lock)."""
        cutoff = now - self._window_seconds
        self._attempts[ip] = [t for t in self._attempts.get(ip, []) if t > cutoff]

    def is_locked_out(self, ip: str) -> bool:
        """Return True if the IP is currently locked out."""
        with self._lock:
            now = time.monotonic()
            self._prune(ip, now)
            return len(self._attempts.get(ip, [])) >= self._max_attempts

    def record_failure(self, ip: str) -> bool:
        """Record a failed attempt and return True if the IP is now locked out."""
        with self._lock:
            now = time.monotonic()
            self._prune(ip, now)
            self._attempts.setdefault(ip, []).append(now)
            return len(self._attempts[ip]) >= self._max_attempts

    def record_success(self, ip: str) -> None:
        """Reset the failed-attempt counter for this IP."""
        with self._lock:
            self._attempts.pop(ip, None)

    def remaining_attempts(self, ip: str) -> int:
        """Return how many failures remain before lockout."""
        with self._lock:
            now = time.monotonic()
            self._prune(ip, now)
            used = len(self._attempts.get(ip, []))
            return max(0, self._max_attempts - used)


# Global login rate-limiter instance
_login_rate_limiter: LoginRateLimiter | None = None
_login_rate_limiter_lock = threading.Lock()


def get_login_rate_limiter() -> LoginRateLimiter:
    """Get the global login rate-limiter instance."""
    global _login_rate_limiter
    if _login_rate_limiter is None:
        with _login_rate_limiter_lock:
            if _login_rate_limiter is None:
                _login_rate_limiter = LoginRateLimiter()
    return _login_rate_limiter


__all__ = [
    "Session",
    "SessionManager",
    "get_session_manager",
    "get_dashboard_password",
    "verify_password",
    "is_password_required",
    "hash_token",
    "SESSION_EXPIRY_SECONDS",
    "LoginRateLimiter",
    "get_login_rate_limiter",
]
