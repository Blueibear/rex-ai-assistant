"""Mobile API route blueprints.

Each module exposes a ``build_*_blueprint`` factory so every app instance
gets independent, injectable views (no module-global request state).
"""

from rex.mobile_api.routes.auth import build_auth_blueprint
from rex.mobile_api.routes.scaffolds import build_scaffolds_blueprint
from rex.mobile_api.routes.status import build_status_blueprint

__all__ = [
    "build_auth_blueprint",
    "build_scaffolds_blueprint",
    "build_status_blueprint",
]
