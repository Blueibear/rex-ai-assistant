"""Explicit authenticated 501 scaffolds for not-yet-implemented surfaces.

Until their real ownership, permission, and persistence contracts are
implemented (Session 2 and later), these routes return HTTP 501 with
``NOT_IMPLEMENTED`` and their capability remains false.  A scaffold never
returns fake data or fake success.
"""

from __future__ import annotations

from typing import Any, NoReturn

from flask import Blueprint

from rex.mobile_api import errors as merr
from rex.mobile_api.auth import require_mobile_auth
from rex.mobile_api.errors import MobileApiError
from rex.mobile_api.services import MobileApiServices

# (rule name, methods, path) — paths from the master spec §6.5.  Chat,
# streaming, voice, and TTS graduated to real routes in Session 2 and are
# registered by their own blueprints.
_SCAFFOLD_ROUTES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("notifications", ("GET",), "/mobile/notifications"),
    ("approvals", ("GET",), "/mobile/approvals"),
    ("tasks", ("GET",), "/mobile/tasks"),
    ("workflows", ("GET",), "/mobile/workflows"),
    ("audit_log", ("GET",), "/mobile/audit-log"),
    ("settings", ("GET",), "/mobile/settings"),
)


def build_scaffolds_blueprint(services: MobileApiServices) -> Blueprint:
    """Build the blueprint registering every explicit 501 scaffold route."""
    bp = Blueprint("mobile_scaffolds", __name__)

    def _make_view(endpoint_name: str) -> Any:
        @require_mobile_auth
        def scaffold_view(**_kwargs: Any) -> NoReturn:
            raise MobileApiError(
                merr.NOT_IMPLEMENTED,
                "This endpoint is not implemented yet.",
                501,
            )

        scaffold_view.__name__ = f"scaffold_{endpoint_name}"
        return scaffold_view

    for name, methods, path in _SCAFFOLD_ROUTES:
        bp.add_url_rule(
            path,
            endpoint=f"scaffold_{name}",
            view_func=_make_view(name),
            methods=list(methods),
        )

    return bp


__all__ = ["build_scaffolds_blueprint"]
