"""OpenClaw tool server blueprint — US-007.

Lightweight Flask Blueprint that exposes Rex's OpenClaw tools as HTTP endpoints,
enabling OpenClaw (and any authorised caller) to invoke Rex's tools via HTTP:

    POST /rex/tools/{tool_name}
    Authorization: Bearer <REX_TOOL_API_KEY>
    Content-Type: application/json

    {"args": {"location": "London"}, "context": {"session_key": "main"}}

Successful response (200)::

    {"status": "success", "result": {...}}

Error response (4xx/5xx)::

    {"error": {"code": "...", "message": "..."}}

Auth
----
The caller must supply ``REX_TOOL_API_KEY`` (environment variable) in every
request via the ``X-API-Key`` header or ``Authorization: Bearer <token>``.

Rate limiting
-------------
A simple deque-based per-key limiter is applied inline.  Limits are
configurable via ``REX_TOOL_RATE_LIMIT`` (requests) and
``REX_TOOL_RATE_WINDOW`` (seconds).  Defaults: 60 requests / 60 seconds.

Policy
------
:class:`~rex.openclaw.policy_adapter.PolicyAdapter` is consulted before
every tool call.  A denied tool returns 403; an approval-required tool also
returns 403 (with a ``"Approval required:"`` prefix).

Typical usage::

    from flask import Flask
    from rex.openclaw.tool_server import ToolServer

    app = Flask(__name__)
    server = ToolServer()
    server.register_all(app)
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from collections import defaultdict, deque
from typing import Any, Callable

from flask import Blueprint, Flask, Response, jsonify, request

from rex.http_errors import (
    BAD_REQUEST,
    FORBIDDEN,
    INTERNAL_ERROR,
    NOT_FOUND,
    TOO_MANY_REQUESTS,
    UNAUTHORIZED,
    error_response,
)
from rex.openclaw.policy_adapter import PolicyAdapter
from rex.openclaw.tool_executor import ApprovalRequiredError, PolicyDeniedError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiting (deque-based, shared across all ToolServer instances)
# ---------------------------------------------------------------------------

_RATE_LIMIT: int = int(os.getenv("REX_TOOL_RATE_LIMIT", "60"))
_RATE_WINDOW: int = int(os.getenv("REX_TOOL_RATE_WINDOW", "60"))
_RATE_CACHE: dict[str, deque] = defaultdict(deque)


def _rate_limit_key() -> str:
    provided = request.headers.get("X-API-Key") or request.headers.get("Authorization")
    if provided:
        token = provided.split()[-1]
        return f"api:{token[:16]}"
    return request.remote_addr or "unknown"


def _check_rate_limit() -> tuple[Response, int] | None:
    """Return 429 error tuple if rate limit exceeded, otherwise None."""
    if _RATE_LIMIT <= 0 or _RATE_WINDOW <= 0:
        return None
    now = time.monotonic()
    key = _rate_limit_key()
    bucket = _RATE_CACHE[key]
    while bucket and now - bucket[0] > _RATE_WINDOW:
        bucket.popleft()
    if len(bucket) >= _RATE_LIMIT:
        return error_response(TOO_MANY_REQUESTS, "Too many requests", 429)
    bucket.append(now)
    return None


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _get_api_key() -> str | None:
    return os.getenv("REX_TOOL_API_KEY") or None


def _extract_request_key() -> str | None:
    provided = request.headers.get("X-API-Key") or request.headers.get("Authorization")
    if not provided:
        return None
    parts = provided.split()
    return parts[-1] if parts else None


def _check_auth() -> tuple[Response, int] | None:
    """Return 401 error tuple if auth fails, otherwise None."""
    api_key = _get_api_key()
    if not api_key:
        return error_response(UNAUTHORIZED, "Tool server API key not configured", 401)
    provided = _extract_request_key()
    if not provided:
        return error_response(UNAUTHORIZED, "Missing API key", 401)
    if not hmac.compare_digest(provided.encode("utf-8"), api_key.encode("utf-8")):
        return error_response(UNAUTHORIZED, "Invalid API key", 401)
    return None


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


def _build_tool_registry() -> dict[str, Callable[..., dict[str, Any]]]:
    """Return a mapping of tool name → callable.

    Imports are deferred so that optional integrations (WooCommerce, Plex,
    etc.) that are not installed do not prevent the server from starting.
    Each import failure is logged at WARNING level and the tool is omitted.
    """
    registry: dict[str, Callable[..., dict[str, Any]]] = {}

    try:
        from rex.openclaw.tools.time_tool import time_now

        registry["time_now"] = time_now
    except ImportError:
        logger.warning("tool_server: time_tool import failed — time_now unavailable")

    try:
        from rex.openclaw.tools.weather_tool import weather_now

        registry["weather_now"] = weather_now
    except ImportError:
        logger.warning("tool_server: weather_tool import failed — weather_now unavailable")

    try:
        from rex.openclaw.tools.email_tool import send_email

        registry["send_email"] = send_email
    except ImportError:
        logger.warning("tool_server: email_tool import failed — send_email unavailable")

    try:
        from rex.openclaw.tools.sms_tool import send_sms

        registry["send_sms"] = send_sms
    except ImportError:
        logger.warning("tool_server: sms_tool import failed — send_sms unavailable")

    try:
        from rex.openclaw.tools.calendar_tool import calendar_create

        registry["calendar_create"] = calendar_create
    except ImportError:
        logger.warning("tool_server: calendar_tool import failed — calendar_create unavailable")

    try:
        from rex.openclaw.tools.ha_tool import ha_call_service

        registry["home_assistant_call_service"] = ha_call_service
    except ImportError:
        logger.warning(
            "tool_server: ha_tool import failed — home_assistant_call_service unavailable"
        )

    try:
        from rex.openclaw.tools.plex_tool import (
            plex_pause,
            plex_play,
            plex_search,
            plex_stop,
        )

        registry["plex_search"] = plex_search
        registry["plex_play"] = plex_play
        registry["plex_pause"] = plex_pause
        registry["plex_stop"] = plex_stop
    except ImportError:
        logger.warning("tool_server: plex_tool import failed — plex_* tools unavailable")

    try:
        from rex.openclaw.tools.wordpress_tool import wp_health_check

        registry["wordpress_health_check"] = wp_health_check
    except ImportError:
        logger.warning(
            "tool_server: wordpress_tool import failed — wordpress_health_check unavailable"
        )

    try:
        from rex.openclaw.tools.woocommerce_tool import (
            wc_create_coupon,
            wc_disable_coupon,
            wc_list_orders,
            wc_list_products,
            wc_set_order_status,
        )

        registry["wc_list_orders"] = wc_list_orders
        registry["wc_list_products"] = wc_list_products
        registry["wc_set_order_status"] = wc_set_order_status
        registry["wc_create_coupon"] = wc_create_coupon
        registry["wc_disable_coupon"] = wc_disable_coupon
    except ImportError:
        logger.warning("tool_server: woocommerce_tool import failed — wc_* tools unavailable")

    return registry


# ---------------------------------------------------------------------------
# ToolServer
# ---------------------------------------------------------------------------


class ToolServer:
    """Lightweight Flask Blueprint exposing Rex's tools as HTTP endpoints.

    Each registered Rex tool is accessible via::

        POST /rex/tools/{tool_name}

    The request body must be JSON with optional ``args`` and ``context`` keys::

        {"args": {"key": "value"}, "context": {"session_key": "main"}}

    Args are unpacked as keyword arguments and passed to the tool handler
    along with the ``context`` dict.  For example, to invoke ``time_now``::

        POST /rex/tools/time_now
        {"args": {"location": "Edinburgh, Scotland"}}

    Auth is required via the ``X-API-Key`` header or
    ``Authorization: Bearer <token>``, where the token is the value of the
    ``REX_TOOL_API_KEY`` environment variable.

    :class:`~rex.openclaw.policy_adapter.PolicyAdapter` is called before
    every tool execution.  A 403 is returned on denial or approval-required.

    Args:
        policy: Optional :class:`~rex.openclaw.policy_adapter.PolicyAdapter`.
            When *None*, the module-level singleton is used.
        tools:  Optional pre-built tool registry (mapping tool name to
            callable).  Inject custom handlers in tests to avoid real I/O.
            When *None*, :func:`_build_tool_registry` is called.
    """

    def __init__(
        self,
        policy: PolicyAdapter | None = None,
        tools: dict[str, Callable[..., dict[str, Any]]] | None = None,
    ) -> None:
        self._policy = policy or PolicyAdapter()
        self._tools: dict[str, Callable[..., dict[str, Any]]] = (
            tools if tools is not None else _build_tool_registry()
        )
        self._blueprint = Blueprint("rex_tools", __name__, url_prefix="/rex/tools")
        self._blueprint.add_url_rule(
            "/<string:tool_name>",
            view_func=self._handle_tool_call,
            methods=["POST"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_all(self, app: Flask) -> None:
        """Register the tool Blueprint on *app*.

        After this call, ``POST /rex/tools/{tool_name}`` is live on the app.

        Args:
            app: Flask application instance.
        """
        app.register_blueprint(self._blueprint)
        logger.info(
            "ToolServer registered %d tool(s): %s",
            len(self._tools),
            ", ".join(sorted(self._tools)),
        )

    # ------------------------------------------------------------------
    # Request handler
    # ------------------------------------------------------------------

    def _handle_tool_call(self, tool_name: str) -> tuple[Response, int]:
        # 1. Rate limit
        rate_err = _check_rate_limit()
        if rate_err is not None:
            return rate_err

        # 2. Auth
        auth_err = _check_auth()
        if auth_err is not None:
            return auth_err

        # 3. Look up tool
        handler = self._tools.get(tool_name)
        if handler is None:
            logger.debug("tool_server: unknown tool %r requested", tool_name)
            return error_response(NOT_FOUND, f"Unknown tool: {tool_name!r}", 404)

        # 4. Parse request body
        body = request.get_json(silent=True) or {}
        args: dict[str, Any] = body.get("args") or {}
        context: dict[str, Any] = body.get("context") or {}

        if not isinstance(args, dict):
            return error_response(BAD_REQUEST, "args must be a JSON object", 400)
        if not isinstance(context, dict):
            return error_response(BAD_REQUEST, "context must be a JSON object", 400)

        # 5. Policy check
        try:
            self._policy.guard(tool_name, metadata=context)
        except PolicyDeniedError as exc:
            logger.warning("tool_server: policy denied tool %r: %s", tool_name, exc)
            return error_response(FORBIDDEN, str(exc), 403)
        except ApprovalRequiredError as exc:
            logger.warning("tool_server: approval required for tool %r: %s", tool_name, exc)
            return error_response(FORBIDDEN, f"Approval required: {exc}", 403)

        # 6. Execute
        try:
            result = handler(**args, context=context)
            logger.debug("tool_server: tool %r succeeded", tool_name)
            return jsonify({"status": "success", "result": result}), 200
        except TypeError as exc:
            logger.warning("tool_server: invalid args for tool %r: %s", tool_name, exc)
            return error_response(BAD_REQUEST, f"Invalid arguments: {exc}", 400)
        except Exception as exc:
            logger.exception("tool_server: tool %r raised unexpected error: %s", tool_name, exc)
            return error_response(INTERNAL_ERROR, "Tool execution failed", 500)
