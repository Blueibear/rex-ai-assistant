"""Flask application factory for the AskRex mobile API gateway (issue #323).

``create_mobile_app`` builds an injectable app with:

- request IDs and privacy-safe request/response logging (no bodies, no tokens);
- ``X-Request-ID`` and ``X-AskRex-API-Version`` headers on every response;
- the nested mobile error envelope (with ``retryable`` and ``request_id``);
- a global JSON body limit (413 before full processing);
- deny-by-default CORS (no origins unless explicitly configured, never ``*``);
- route-specific rate limiting (login/refresh have their own limits);
- only ``/mobile/*`` routes — the Electron GUI/admin server is a separate app;
- fail-closed TLS resolution for non-loopback binds (S7): ``services.tls_material``
  and ``app.extensions["mobile_api_tls"]`` are None on loopback dev binds and a
  provisioned :class:`rex.mobile_api.tls.TlsMaterial` otherwise. Building the
  app raises ``MobileTlsConfigurationError`` when a non-loopback bind cannot
  get usable TLS material.

Importing this module has no side effects; database migrations run when the
factory is called.  In-memory rate-limit storage is suitable only for the
single-process development server and is documented as such.
"""

from __future__ import annotations

import logging
import math

from flask import Flask, Response, g, request

from rex.config import MobileApiConfig
from rex.mobile_api import errors as merr
from rex.mobile_api.db import migrate_users_db
from rex.mobile_api.errors import install_mobile_error_handlers
from rex.mobile_api.routes import (
    build_auth_blueprint,
    build_chat_blueprint,
    build_home_blueprint,
    build_pairing_blueprint,
    build_scaffolds_blueprint,
    build_status_blueprint,
    build_strong_auth_blueprint,
    build_voice_blueprint,
)
from rex.mobile_api.services import MobileApiServices
from rex.mobile_api.tls import MobileTlsConfigurationError, host_is_loopback
from rex.mobile_api.websocket import register_websocket

logger = logging.getLogger(__name__)


def _resolve_config(config: MobileApiConfig | None) -> MobileApiConfig:
    if config is not None:
        return config
    try:
        from rex.config import settings  # noqa: PLC0415

        resolved = getattr(settings, "mobile_api", None)
        if isinstance(resolved, MobileApiConfig):
            return resolved
    except Exception:  # pragma: no cover - config loading edge cases
        logger.warning("Falling back to default mobile_api configuration")
    return MobileApiConfig()


def _install_rate_limiter(app: Flask, config: MobileApiConfig):
    from flask import jsonify  # noqa: PLC0415
    from flask_limiter import Limiter, RequestLimit  # noqa: PLC0415
    from flask_limiter.util import get_remote_address  # noqa: PLC0415

    def _on_breach(request_limit: RequestLimit):
        import time as _time  # noqa: PLC0415

        reset_at = getattr(request_limit, "reset_at", None)
        if reset_at is not None:
            retry_after = max(1, math.ceil(reset_at - _time.time()))
        else:
            retry_after = 60
        body = {
            "error": {
                "code": merr.RATE_LIMITED,
                "message": "Too many requests. Please slow down.",
                "retryable": True,
                "request_id": getattr(g, "request_id", None),
            }
        }
        response = jsonify(body)
        response.status_code = 429
        response.headers["Retry-After"] = str(retry_after)
        return response

    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=[config.rate_limit_default],
        # In-memory storage: single-process local development only.  A shared
        # storage backend is required before any multi-process deployment.
        storage_uri="memory://",
        headers_enabled=True,
        on_breach=_on_breach,
    )
    return limiter


def _install_cors(app: Flask, config: MobileApiConfig) -> None:
    """Deny-by-default CORS: only explicitly configured origins, never '*'."""
    if not config.allowed_origins:
        return
    from flask_cors import CORS  # noqa: PLC0415

    CORS(
        app,
        resources={r"/mobile/*": {"origins": list(config.allowed_origins)}},
        supports_credentials=False,
    )


def create_mobile_app(
    config: MobileApiConfig | None = None,
    services: MobileApiServices | None = None,
) -> Flask:
    """Create a configured mobile API Flask application.

    Args:
        config: Typed mobile API configuration.  Defaults to the global
            ``settings.mobile_api`` group, then safe library defaults.
        services: Pre-built service container (tests inject temporary
            database paths, fake clocks, and deterministic generators).

    Raises:
        MobileAuthConfigurationError: When ``REX_JWT_SECRET`` is missing or
            too weak — the auth service fails closed before serving.
        MobileTlsConfigurationError: When the configured bind requires TLS but
            the injected/default service container has no usable TLS material.
    """
    if services is None:
        services = MobileApiServices.build(_resolve_config(config))
    cfg = services.config
    if (cfg.require_tls or not host_is_loopback(cfg.host)) and services.tls_material is None:
        raise MobileTlsConfigurationError(
            "Secure TLS material is required for this mobile gateway configuration."
        )

    # Idempotent canonical users.db migration (sessions/refresh tables).
    migrate_users_db(services.db_path)

    app = Flask("rex_mobile_api")
    # The transport-level cap must admit multipart voice uploads (15 MiB by
    # default); JSON routes enforce the tighter ``max_json_bytes`` limit in
    # ``parse_json_body`` before any parsing work.
    app.config["MAX_CONTENT_LENGTH"] = max(cfg.max_json_bytes, cfg.max_audio_bytes + 128 * 1024)
    app.extensions["mobile_api_services"] = services
    # None on a loopback dev bind; a resolved TlsMaterial (S7) whenever cfg.host
    # is non-loopback or cfg.require_tls opts a loopback bind into TLS too.
    app.extensions["mobile_api_tls"] = services.tls_material

    # Reused canonical middleware: assigns g.request_id before authentication
    # and logs method/path/status only — request and response bodies, tokens,
    # and passwords are never logged.
    from rex.request_logging import install_request_logging  # noqa: PLC0415

    install_request_logging(app)
    install_mobile_error_handlers(app)

    @app.before_request
    def _enforce_owned_tls_transport() -> None:
        # The supported LAN topology terminates TLS inside this process. If an
        # operator accidentally serves the Flask app over plaintext through a
        # different WSGI runner, fail closed rather than exposing authenticated
        # mobile traffic on an unencrypted socket.
        if services.tls_material is not None and not request.is_secure:
            raise merr.MobileApiError(
                merr.TLS_REQUIRED,
                "Secure HTTPS transport is required for this mobile gateway.",
                426,
            )

    _install_cors(app, cfg)
    limiter = _install_rate_limiter(app, cfg)
    app.extensions["mobile_api_limiter"] = limiter

    @app.after_request
    def _add_mobile_headers(response: Response) -> Response:
        request_id = getattr(g, "request_id", None)
        if request_id:
            response.headers["X-Request-ID"] = request_id
        response.headers["X-AskRex-API-Version"] = cfg.api_version
        return response

    app.register_blueprint(build_status_blueprint(services))
    app.register_blueprint(build_auth_blueprint(services, limiter))
    app.register_blueprint(build_pairing_blueprint(services, limiter))
    app.register_blueprint(build_strong_auth_blueprint(services))
    app.register_blueprint(build_home_blueprint(services))
    app.register_blueprint(build_chat_blueprint(services, limiter))
    app.register_blueprint(build_voice_blueprint(services, limiter))
    app.register_blueprint(build_scaffolds_blueprint(services))

    # WebSocket /mobile/chat/stream — registered only when the validated
    # Flask-Sock stack is installed; the capability stays false otherwise.
    ws_registered = register_websocket(app, services)
    services.websocket_registered = ws_registered

    logger.info(
        "Mobile API app created (api_version=%s, websocket=%s)",
        cfg.api_version,
        "on" if ws_registered else "off",
    )
    return app


__all__ = ["create_mobile_app"]
