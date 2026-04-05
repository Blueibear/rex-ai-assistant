"""API key validation for Rex Flask endpoints.

Provides a reusable decorator and validator that:
- Reads the expected key from an environment variable
- Performs a constant-time comparison to prevent timing attacks
- Rejects unauthorized requests with a 401 response
- Logs every failed authentication attempt
"""

from __future__ import annotations

import functools
import hmac
import logging
import os
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Header names checked in order
_KEY_HEADERS = ("X-API-Key", "Authorization")


def _extract_key_from_request(headers: Any) -> str | None:
    """Extract an API key from request headers.

    Checks X-API-Key first, then Authorization (strips 'Bearer'/'Token' prefix).

    Args:
        headers: A Flask-compatible headers mapping.

    Returns:
        The raw key string or None if not present.
    """
    for header in _KEY_HEADERS:
        value = headers.get(header)
        if value:
            parts = value.split()
            return parts[-1] if parts else None
    return None


def validate_api_key(provided: str | None, expected: str | None) -> bool:
    """Validate a provided API key against the expected value.

    Uses constant-time comparison (``hmac.compare_digest``) to prevent
    timing side-channel attacks.

    Args:
        provided: The key supplied by the caller.
        expected: The correct key to compare against.

    Returns:
        True if both are non-empty and equal, False otherwise.
    """
    if not provided or not expected:
        return False
    try:
        return hmac.compare_digest(
            provided.encode("utf-8"),
            expected.encode("utf-8"),
        )
    except Exception:
        return False


class ApiKeyValidator:
    """Configurable API key validator with logging support.

    Reads the expected key from an environment variable. If no key is
    configured the validator rejects all requests.

    Args:
        env_var: Name of the environment variable holding the expected key.
        label: Human-readable label used in log messages (e.g. "speak API").
    """

    def __init__(self, env_var: str = "REX_API_KEY", label: str = "API") -> None:
        self._env_var = env_var
        self._label = label

    def get_expected_key(self) -> str | None:
        """Return the currently configured expected key (or None)."""
        return os.getenv(self._env_var) or None

    def check(self, provided: str | None, *, remote_addr: str = "unknown") -> bool:
        """Check a provided key and log failures.

        Args:
            provided: Key extracted from the incoming request.
            remote_addr: Client address for log context.

        Returns:
            True if the key is valid, False otherwise.
        """
        expected = self.get_expected_key()

        if not expected:
            logger.warning(
                "[%s] No API key configured (%s is unset) — rejecting request from %s",
                self._label,
                self._env_var,
                remote_addr,
            )
            return False

        if not validate_api_key(provided, expected):
            logger.warning(
                "[%s] Invalid API key from %s — unauthorized",
                self._label,
                remote_addr,
            )
            return False

        return True


# Default shared validator — routes that need API key auth import this.
_default_validator = ApiKeyValidator(env_var="REX_API_KEY", label="API")


def require_api_key(
    func: Callable | None = None,
    *,
    env_var: str = "REX_API_KEY",
    label: str = "API",
) -> Any:
    """Flask route decorator that enforces API key authentication.

    Reads the expected key from *env_var* and compares it to the value
    supplied in the ``X-API-Key`` or ``Authorization`` header.  Returns
    HTTP 401 with a JSON error body when the key is absent or incorrect.
    All failures are logged at WARNING level.

    Usage::

        @app.route("/protected")
        @require_api_key
        def protected_route():
            ...

        # Custom env var:
        @app.route("/other")
        @require_api_key(env_var="REX_SPEAK_API_KEY", label="speak")
        def other_route():
            ...

    Args:
        func: The view function to wrap (when used without arguments).
        env_var: Environment variable name for the expected key.
        label: Label used in log messages.
    """

    # Support both @require_api_key and @require_api_key(env_var=...)
    def decorator(view_fn: Callable) -> Callable:
        validator = ApiKeyValidator(env_var=env_var, label=label)

        @functools.wraps(view_fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Import Flask lazily so this module can be imported without Flask
            from flask import request  # noqa: PLC0415

            provided = _extract_key_from_request(request.headers)
            remote = request.remote_addr or "unknown"

            if not validator.check(provided, remote_addr=remote):
                from rex.http_errors import UNAUTHORIZED, error_response  # noqa: PLC0415

                return error_response(UNAUTHORIZED, "Unauthorized", 401)

            return view_fn(*args, **kwargs)

        return wrapper

    if func is not None:
        # Used as @require_api_key (no parentheses)
        return decorator(func)

    # Used as @require_api_key(...) with arguments
    return decorator


__all__ = [
    "ApiKeyValidator",
    "require_api_key",
    "validate_api_key",
]
