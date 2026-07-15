"""Public status and capabilities endpoints.

These may be unauthenticated but reveal only non-sensitive compatibility and
health data — no secrets, paths, account IDs, usernames, or tokens.
"""

from __future__ import annotations

from typing import Any

from flask import Blueprint, g, jsonify

from rex.mobile_api.capabilities import capabilities_payload, server_version
from rex.mobile_api.services import MobileApiServices


def build_status_blueprint(services: MobileApiServices) -> Blueprint:
    """Build the ``/mobile/status`` and ``/mobile/capabilities`` blueprint."""
    bp = Blueprint("mobile_status", __name__, url_prefix="/mobile")

    @bp.get("/status")
    def status() -> Any:
        return jsonify(
            {
                "status": "ok",
                "api_version": services.config.api_version,
                "server_version": server_version(),
                "request_id": getattr(g, "request_id", None),
            }
        )

    @bp.get("/capabilities")
    def capabilities() -> Any:
        return jsonify(capabilities_payload(services.config, services))

    return bp


__all__ = ["build_status_blueprint"]
