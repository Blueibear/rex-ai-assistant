"""Request and response logging middleware for Flask applications.

Installs ``before_request`` / ``after_request`` hooks on a Flask app that log:

- **Request**: method, path, anonymized client IP, timestamp, request ID.
- **Response**: status code, duration in milliseconds, request ID.

Request body and response body are **never** logged to avoid PII leakage.

A unique ``request_id`` (UUID4) is generated per request and stored in
``flask.g`` so that downstream code can include it in application-level logs
for correlation.

Usage::

    from rex.request_logging import install_request_logging

    app = Flask(__name__)
    install_request_logging(app)

IP anonymisation
----------------
By default the last octet of IPv4 addresses (and the last 64 bits of IPv6
addresses) are replaced with ``0`` so that no individual can be identified
from the logs.  Full IP logging can be enabled by setting
``REX_LOG_FULL_IP=1`` in the environment.
"""

from __future__ import annotations

import logging
import os
import re
import time
import uuid

from flask import Flask, Response, g, request

logger = logging.getLogger("rex.request_logging")

# Regex that matches the last octet of an IPv4 address.
_IPV4_LAST_OCTET = re.compile(r"\.\d+$")
# Regex that matches the last 64 bits (4 groups) of an IPv6 address.
_IPV6_LAST_HALF = re.compile(r"(:[0-9a-fA-F]*){1,4}$")


def _anonymize_ip(ip: str) -> str:
    """Replace the identifying portion of *ip* with zeroes.

    - IPv4: replaces the last octet  (``192.168.1.42`` → ``192.168.1.0``)
    - IPv6: replaces the last 64 bits (``2001:db8::1234:5678`` → ``2001:db8::0``)
    - Unrecognised format: returned unchanged.
    """
    if not ip:
        return "unknown"
    if ":" in ip:
        # IPv6
        return _IPV6_LAST_HALF.sub(":0", ip)
    if "." in ip:
        # IPv4
        return _IPV4_LAST_OCTET.sub(".0", ip)
    return ip


def _log_full_ip() -> bool:
    """Return True when full IP logging is enabled via environment."""
    return os.environ.get("REX_LOG_FULL_IP", "").lower() in ("1", "true", "yes")


_INSTALLED_KEY = "rex_request_logging_installed"


def install_request_logging(app: Flask) -> None:
    """Install request/response logging hooks on *app*.

    The hooks are idempotent — calling this function multiple times on the
    same app instance will not duplicate log entries.

    Args:
        app: The :class:`flask.Flask` application to instrument.
    """
    if app.extensions.get(_INSTALLED_KEY):
        return
    app.extensions[_INSTALLED_KEY] = True

    @app.before_request
    def _log_request() -> None:
        """Assign a request ID and log the incoming request."""
        g.request_id = str(uuid.uuid4())
        g.request_start = time.monotonic()

        raw_ip = request.remote_addr or "unknown"
        client_ip = raw_ip if _log_full_ip() else _anonymize_ip(raw_ip)

        logger.info(
            "REQUEST  %s %s from %s",
            request.method,
            request.path,
            client_ip,
            extra={
                "request_id": g.request_id,
                "method": request.method,
                "path": request.path,
                "client_ip": client_ip,
            },
        )

    @app.after_request
    def _log_response(response: Response) -> Response:
        """Log the outgoing response with status code and duration."""
        request_id = getattr(g, "request_id", "n/a")
        start = getattr(g, "request_start", None)
        duration_ms = int((time.monotonic() - start) * 1000) if start is not None else -1

        logger.info(
            "RESPONSE %s %s → %d (%dms)",
            request.method,
            request.path,
            response.status_code,
            duration_ms,
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response


__all__ = [
    "install_request_logging",
    "_anonymize_ip",
    "_log_full_ip",
]
