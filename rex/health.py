"""Liveness and readiness health check endpoints for Rex Flask applications.

Provides a Flask Blueprint with two endpoints:

- ``GET /health/live``   — Always 200 while the process is running.
- ``GET /health/ready``  — 200 when all registered dependency checks pass;
                           503 with a JSON body listing failing checks otherwise.

Usage::

    from rex.health import create_health_blueprint, check_config

    app = Flask(__name__)
    app.register_blueprint(create_health_blueprint(checks=[check_config]))

Custom checks can be added as callables::

    def my_check() -> str | None:
        # Return None (or True) to indicate healthy.
        # Return a non-empty string to indicate the failure reason.
        try:
            db.ping()
            return None
        except Exception as exc:
            return str(exc)

    bp = create_health_blueprint(checks=[check_config, my_check])

Note: ``check_dashboard_db`` was removed as part of the OpenClaw migration.
Dashboard storage health is now the responsibility of the OpenClaw layer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from flask import Blueprint, jsonify

logger = logging.getLogger("rex.health")

# Type for a readiness check: callable with no args that returns None/empty
# string on success, or a non-empty string describing the failure.
ReadinessCheck = Callable[[], "str | None"]


# ---------------------------------------------------------------------------
# Built-in dependency checks
# ---------------------------------------------------------------------------


def check_config() -> str | None:
    """Return None if config loads cleanly, or an error string if it fails."""
    try:
        from rex.config import load_config

        load_config()
        return None
    except Exception as exc:  # noqa: BLE001
        return f"config: {exc}"


# ---------------------------------------------------------------------------
# Blueprint factory
# ---------------------------------------------------------------------------


def create_health_blueprint(
    checks: list[ReadinessCheck] | None = None,
    url_prefix: str = "",
) -> Blueprint:
    """Create a Flask Blueprint with ``/health/live`` and ``/health/ready`` routes.

    Args:
        checks: List of callables to run for the readiness check.  Each callable
            takes no arguments and returns ``None`` (healthy) or a non-empty
            string describing the failure.  When ``None``, no readiness checks
            are run and ``/health/ready`` always returns 200.
        url_prefix: Optional URL prefix for the blueprint.  Default: ``""``.

    Returns:
        A :class:`flask.Blueprint` ready to be registered on an app.
    """
    bp = Blueprint("health", __name__, url_prefix=url_prefix)
    _checks: list[ReadinessCheck] = list(checks) if checks else []

    @bp.route("/health/live", methods=["GET"])
    def liveness():
        """Liveness probe — always 200 while the process is running."""
        return jsonify({"status": "ok"}), 200

    @bp.route("/health/ready", methods=["GET"])
    def readiness():
        """Readiness probe — 200 when all dependency checks pass, 503 otherwise."""
        unavailable: dict[str, str] = {}
        for check in _checks:
            try:
                result = check()
                if result:
                    # Non-empty string → failure; use the check function name as key.
                    key = getattr(check, "__name__", repr(check))
                    unavailable[key] = result
            except Exception as exc:  # noqa: BLE001
                key = getattr(check, "__name__", repr(check))
                unavailable[key] = str(exc)
                logger.warning("Health check %r raised: %s", key, exc)

        if unavailable:
            return (
                jsonify({"status": "unavailable", "unavailable": unavailable}),
                503,
            )
        return jsonify({"status": "ready"}), 200

    return bp


__all__ = [
    "create_health_blueprint",
    "check_config",
    "ReadinessCheck",
]
