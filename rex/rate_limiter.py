"""Per-IP rate limiting for public-facing Flask endpoints.

Installs a Flask-Limiter based rate limiter that:

- Applies a configurable default limit to all routes (default: 60/minute).
- Exempts ``/health/`` endpoints from rate limiting.
- Returns a 429 response using the standard error envelope with a
  ``Retry-After`` header when the limit is exceeded.

Environment variables
---------------------
API_RATE_LIMIT   Rate limit string (default: "60 per minute").
                 Accepts any Flask-Limiter limit string, e.g. "120 per hour".

Usage::

    from rex.rate_limiter import install_rate_limiter

    app = Flask(__name__)
    install_rate_limiter(app)
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)

_INSTALLED_KEY = "rex_rate_limiter_installed"
_DEFAULT_LIMIT = "60 per minute"
_HEALTH_PREFIX = "/health/"
_HEALTH_EXACT = {"/health/live", "/health/ready"}


def _is_health_endpoint() -> bool:
    """Return True when the current request is for a health-check endpoint."""
    from flask import request  # noqa: PLC0415

    path = request.path
    return path in _HEALTH_EXACT or path.startswith(_HEALTH_PREFIX)


def _read_limit() -> str:
    """Read the rate limit string from the environment, falling back to default."""
    raw = os.environ.get("API_RATE_LIMIT", "").strip()
    if raw:
        return raw
    return _DEFAULT_LIMIT


def install_rate_limiter(app: Any, limit_str: str | None = None) -> Any:
    """Install a per-IP rate limiter on *app*.

    The limiter is idempotent — calling this function more than once on the
    same app instance is safe and will skip re-installation.

    Args:
        app: The :class:`flask.Flask` application to instrument.
        limit_str: Override for the rate limit string.  When ``None`` (default),
            the value is read from the ``API_RATE_LIMIT`` environment variable,
            falling back to ``"60 per minute"``.

    Returns:
        The :class:`flask_limiter.Limiter` instance that was installed, or the
        previously installed instance if already present.
    """
    if app.extensions.get(_INSTALLED_KEY):
        return app.extensions[_INSTALLED_KEY]

    from flask import jsonify  # noqa: PLC0415
    from flask_limiter import Limiter, RequestLimit  # noqa: PLC0415
    from flask_limiter.util import get_remote_address  # noqa: PLC0415

    from rex.http_errors import TOO_MANY_REQUESTS  # noqa: PLC0415

    effective_limit = limit_str if limit_str is not None else _read_limit()

    def _on_breach(request_limit: RequestLimit) -> Any:
        import time as _time  # noqa: PLC0415

        reset_at = getattr(request_limit, "reset_at", None)
        if reset_at is not None:
            # reset_at is a Unix timestamp (int or float)
            remaining = reset_at - _time.time()
            retry_after_seconds = max(1, math.ceil(remaining))
        else:
            retry_after_seconds = 60
        body = {
            "error": {
                "code": TOO_MANY_REQUESTS,
                "message": "Too many requests. Please slow down.",
            }
        }
        resp = jsonify(body)
        resp.status_code = 429
        resp.headers["Retry-After"] = str(max(1, retry_after_seconds))
        return resp

    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=[effective_limit],
        default_limits_exempt_when=_is_health_endpoint,
        storage_uri="memory://",
        headers_enabled=True,
        on_breach=_on_breach,
    )

    app.extensions[_INSTALLED_KEY] = limiter

    logger.info(
        "Rate limiter installed: limit=%r (health endpoints exempt)",
        effective_limit,
    )
    return limiter


__all__ = [
    "install_rate_limiter",
]
