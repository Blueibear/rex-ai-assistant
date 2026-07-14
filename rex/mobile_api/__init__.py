"""Authenticated mobile API gateway for AskRex (issue #323).

This package owns the mobile HTTP transport and security boundary:
access-token validation, rotating refresh sessions, request IDs, structured
errors, rate limits, body limits, and truthful capabilities.

It deliberately does NOT own a second assistant, user directory, permission
model, memory implementation, or policy engine — it reuses the canonical Rex
services and ``data/users.db``.

Importing this package must have no side effects (no listeners, no database
mutation, no model loading, no secret reads).  Use
:func:`rex.mobile_api.app.create_mobile_app` to build a configured Flask app.
"""

__all__ = ["create_mobile_app"]


def create_mobile_app(*args, **kwargs):
    """Lazily import and delegate to :func:`rex.mobile_api.app.create_mobile_app`."""
    from rex.mobile_api.app import create_mobile_app as _factory

    return _factory(*args, **kwargs)
