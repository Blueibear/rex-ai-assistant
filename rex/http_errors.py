"""Standard HTTP error response helpers.

All API error responses must use the envelope returned by ``error_response``
so that clients can parse errors without special-casing each endpoint.

Envelope shape::

    {"error": {"code": "<CODE>", "message": "<human-readable text>"}}

For 500-level errors the envelope also includes a ``request_id`` field::

    {"error": {"code": "<CODE>", "message": "...", "request_id": "<UUID>"}}
"""

from __future__ import annotations

import logging
from typing import Any

# Standard error code constants
BAD_REQUEST = "BAD_REQUEST"
UNAUTHORIZED = "UNAUTHORIZED"
FORBIDDEN = "FORBIDDEN"
NOT_FOUND = "NOT_FOUND"
PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
TOO_MANY_REQUESTS = "TOO_MANY_REQUESTS"
INTERNAL_ERROR = "INTERNAL_ERROR"
SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
UNPROCESSABLE = "UNPROCESSABLE"

# HTTP status → error code mapping for werkzeug HTTPException handling
_HTTP_STATUS_TO_CODE: dict[int, str] = {
    400: BAD_REQUEST,
    401: UNAUTHORIZED,
    403: FORBIDDEN,
    404: NOT_FOUND,
    405: BAD_REQUEST,
    409: BAD_REQUEST,
    410: NOT_FOUND,
    422: UNPROCESSABLE,
    429: TOO_MANY_REQUESTS,
    503: SERVICE_UNAVAILABLE,
}

_INSTALLED_KEY = "rex_error_envelope_installed"

logger = logging.getLogger(__name__)


def error_response(
    code: str,
    message: str,
    http_status: int,
    *,
    request_id: str | None = None,
) -> tuple[Any, int]:
    """Return a Flask ``(Response, status)`` tuple with the standard error envelope.

    Args:
        code: Machine-readable error code string (e.g. ``"BAD_REQUEST"``).
        message: Human-readable description of the error.
        http_status: HTTP status code (e.g. 400, 401, 500).
        request_id: Optional request ID for log correlation.  When supplied,
            included as ``error.request_id`` in the envelope.

    Returns:
        A ``(jsonified_response, http_status)`` tuple suitable for returning
        directly from a Flask view function.
    """
    from flask import jsonify  # noqa: PLC0415 — lazy import so module works without Flask

    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if request_id is not None:
        body["error"]["request_id"] = request_id
    return jsonify(body), http_status


def install_error_envelope_handler(app: Any) -> None:
    """Install centralised error-envelope handlers on *app*.

    Registers two Flask error handlers so that error formatting lives in one
    place rather than being duplicated per endpoint:

    1. ``werkzeug.exceptions.HTTPException`` — maps 4xx/5xx HTTP exceptions to
       the standard envelope.
    2. ``Exception`` — catches all unhandled exceptions as a 500 response and
       includes ``error.request_id`` from ``flask.g`` for log correlation.

    The installation is idempotent; calling this function more than once on the
    same app instance is safe.

    Args:
        app: The :class:`flask.Flask` application to instrument.
    """
    if app.extensions.get(_INSTALLED_KEY):
        return
    app.extensions[_INSTALLED_KEY] = True

    from werkzeug.exceptions import HTTPException

    @app.errorhandler(HTTPException)
    def _handle_http_exception(exc: HTTPException) -> tuple[Any, int]:
        status = exc.code or 500
        code = _HTTP_STATUS_TO_CODE.get(status, INTERNAL_ERROR if status >= 500 else BAD_REQUEST)
        message = exc.description or exc.name or "An error occurred."
        return error_response(code, message, status)

    @app.errorhandler(Exception)
    def _handle_unexpected_exception(exc: Exception) -> tuple[Any, int]:
        from flask import g  # noqa: PLC0415

        request_id: str | None = getattr(g, "request_id", None)
        logger.exception("Unhandled exception (request_id=%s): %s", request_id, exc)
        return error_response(
            INTERNAL_ERROR,
            "An unexpected error occurred.",
            500,
            request_id=request_id,
        )


__all__ = [
    "BAD_REQUEST",
    "FORBIDDEN",
    "INTERNAL_ERROR",
    "NOT_FOUND",
    "PAYLOAD_TOO_LARGE",
    "SERVICE_UNAVAILABLE",
    "TOO_MANY_REQUESTS",
    "UNAUTHORIZED",
    "UNPROCESSABLE",
    "error_response",
    "install_error_envelope_handler",
]
