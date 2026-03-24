"""Production configuration defaults for Rex Flask applications.

Detects production mode from the ``ENVIRONMENT`` environment variable and
applies safe defaults to a Flask application instance:

- ``app.debug`` forced to ``False``
- ``app.testing`` forced to ``False``
- A generic JSON 500 error handler registered so that unhandled exception
  details (stack traces) are never returned to API clients
- Development-only routes (prefix ``/debug``) blocked with 404

Usage::

    from rex.production_config import apply_production_defaults, is_production

    app = Flask(__name__)
    apply_production_defaults(app)  # safe to call in every environment
"""

from __future__ import annotations

import os

from flask import Flask, Response, abort, request

from rex.http_errors import INTERNAL_ERROR, error_response

# Prefixes considered development-only and blocked in production.
_DEV_ONLY_PREFIXES: frozenset[str] = frozenset({"/debug"})

_INSTALLED_KEY = "rex_production_config_installed"


def is_production() -> bool:
    """Return ``True`` when the application is running in production mode.

    Production mode is activated by setting the ``ENVIRONMENT`` environment
    variable to ``production`` (case-insensitive).  Any other value (or the
    variable being unset) is treated as a non-production environment.

    Returns:
        ``True`` if ``ENVIRONMENT=production``, ``False`` otherwise.
    """
    return os.getenv("ENVIRONMENT", "development").lower() == "production"


def apply_production_defaults(app: Flask) -> None:
    """Apply safe production defaults to *app*.

    This function is idempotent — calling it multiple times on the same app
    instance has no additional effect.

    When :func:`is_production` returns ``True`` the following are applied:

    - ``app.debug`` set to ``False``
    - ``app.testing`` set to ``False``
    - A ``500`` error handler that returns a generic JSON envelope without any
      traceback or internal exception details
    - A ``before_request`` hook that returns ``404`` for any path whose prefix
      appears in ``_DEV_ONLY_PREFIXES`` (e.g. ``/debug``)

    When :func:`is_production` returns ``False`` the function returns without
    modifying the app, so development behaviour is unchanged.

    Args:
        app: The :class:`flask.Flask` application to configure.
    """
    if app.extensions.get(_INSTALLED_KEY):
        return
    app.extensions[_INSTALLED_KEY] = True

    if not is_production():
        return

    # Disable debug and testing modes explicitly.
    app.debug = False
    app.testing = False

    @app.errorhandler(500)
    def _handle_500(exc: Exception) -> tuple[Response, int]:
        """Return a generic error envelope without traceback details."""
        return error_response(INTERNAL_ERROR, "An unexpected error occurred.", 500)

    @app.before_request
    def _block_dev_routes() -> None:
        """Return 404 for development-only route prefixes."""
        if any(request.path.startswith(prefix) for prefix in _DEV_ONLY_PREFIXES):
            abort(404)


__all__ = [
    "is_production",
    "apply_production_defaults",
    "_DEV_ONLY_PREFIXES",
]
