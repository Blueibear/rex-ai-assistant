"""Standard HTTP error response helpers.

All API error responses must use the envelope returned by ``error_response``
so that clients can parse errors without special-casing each endpoint.

Envelope shape::

    {"error": {"code": "<CODE>", "message": "<human-readable text>"}}
"""

from __future__ import annotations

from typing import Any

# Standard error code constants
BAD_REQUEST = "BAD_REQUEST"
UNAUTHORIZED = "UNAUTHORIZED"
FORBIDDEN = "FORBIDDEN"
NOT_FOUND = "NOT_FOUND"
TOO_MANY_REQUESTS = "TOO_MANY_REQUESTS"
INTERNAL_ERROR = "INTERNAL_ERROR"
SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
UNPROCESSABLE = "UNPROCESSABLE"


def error_response(
    code: str,
    message: str,
    http_status: int,
) -> tuple[Any, int]:
    """Return a Flask ``(Response, status)`` tuple with the standard error envelope.

    Args:
        code: Machine-readable error code string (e.g. ``"BAD_REQUEST"``).
        message: Human-readable description of the error.
        http_status: HTTP status code (e.g. 400, 401, 500).

    Returns:
        A ``(jsonified_response, http_status)`` tuple suitable for returning
        directly from a Flask view function.
    """
    from flask import jsonify  # noqa: PLC0415 — lazy import so module works without Flask

    body = {"error": {"code": code, "message": message}}
    return jsonify(body), http_status


__all__ = [
    "BAD_REQUEST",
    "FORBIDDEN",
    "INTERNAL_ERROR",
    "NOT_FOUND",
    "SERVICE_UNAVAILABLE",
    "TOO_MANY_REQUESTS",
    "UNAUTHORIZED",
    "UNPROCESSABLE",
    "error_response",
]
