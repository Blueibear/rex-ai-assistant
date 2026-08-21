"""OpenClaw tool bridge — US-P4-003 (updated US-P7-008, US-009).

Implements :class:`~rex.contracts.tool_routing.ToolRoutingProtocol` by
delegating to :mod:`rex.openclaw.tool_executor` module-level functions.

This bridge presents the ``ToolRoutingProtocol`` interface so that callers
do not need to import the internal tool executor directly.

When ``use_openclaw_tools`` is True and the OpenClaw gateway is configured,
:meth:`execute_tool` may use the legacy direct HTTP path only for read-only
operations. Mutations always enter Rex's canonical executor/dispatcher so
permissions, confirmation, action lifecycle, reconnect authority, and
verification remain authoritative. Read failures may fall back locally; a
403 response raises :class:`~rex.openclaw.tool_executor.PolicyDeniedError`.

When the flag is False or no gateway URL is set, all calls go through the
local :func:`~rex.openclaw.tool_executor.execute_tool` — identical
behaviour to the pre-HTTP era.

Typical usage::

    from rex.openclaw.tool_bridge import ToolBridge

    bridge = ToolBridge()

    # Parse an LLM output line
    req = bridge.parse_tool_request('TOOL_REQUEST: {"tool": "time_now", "args": {}}')

    # Execute a decoded request
    result = bridge.execute_tool(req, {})

    # Full round-trip: detect → execute → re-call model
    final_text = bridge.route_if_tool_request(llm_output, context, model_fn)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from rex.openclaw.errors import (
    OpenClawAPIError,
    OpenClawAuthError,
    OpenClawConnectionError,
    OpenClawOutcomeUnknownError,
    OpenClawUnavailableError,
)
from rex.openclaw.http_client import get_openclaw_client
from rex.openclaw.tool_executor import (
    PolicyDeniedError,
)
from rex.openclaw.tool_executor import (
    execute_tool as _execute_tool,
)
from rex.openclaw.tool_executor import (
    parse_tool_request as _parse_tool_request,
)
from rex.openclaw.tool_executor import (
    route_if_tool_request as _route_if_tool_request,
)

if TYPE_CHECKING:
    from rex.config import AppConfig

logger = logging.getLogger(__name__)

_LEGACY_DIRECT_READ_TOOLS = frozenset({"time_now", "weather_now", "web_search"})


def _legacy_http_operation(tool_name: str) -> str:
    """Resolve operation authority; unknown legacy HTTP tools fail closed as mutations."""
    try:
        from rex.tools.registry import get_default_registry

        tool = get_default_registry().get(tool_name)
    except Exception:
        tool = None
    if tool is not None:
        return tool.operation
    return "read" if tool_name in _LEGACY_DIRECT_READ_TOOLS else "mutation"


def _warn_gateway_fallback(tool_name: str, exc: Exception, *, failure: str | None = None) -> None:
    """Emit a machine-readable warning before local fallback."""
    failure_name = failure or type(exc).__name__
    logger.warning(
        "OpenClaw gateway unavailable for tool=%s; falling back to local execution: %s",
        tool_name,
        exc,
        extra={
            "event": "openclaw.tool_fallback",
            "tool_name": tool_name,
            "failure": failure_name,
        },
    )


class ToolBridge:
    """Adapter that presents Rex's tool executor as an OpenClaw tool provider.

    Implements :class:`~rex.contracts.tool_routing.ToolRoutingProtocol` by
    delegating all three core operations to the corresponding module-level
    functions in :mod:`rex.openclaw.tool_executor`.

    When ``config.use_openclaw_tools`` is True and the gateway is reachable,
    :meth:`execute_tool` may use direct HTTP for read-only compatibility calls.
    Mutations remain on Rex's canonical executor/dispatcher path. All other
    methods always run locally.

    Args:
        config: Optional :class:`~rex.config.AppConfig`.  When *None*, the
            config is loaded lazily from ``rex_config.json`` on first use.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # ToolRoutingProtocol implementation
    # ------------------------------------------------------------------

    def parse_tool_request(self, text: str) -> dict[str, Any] | None:
        """Return a parsed tool-request dict or *None* if not a tool call.

        Delegates to :func:`~rex.openclaw.tool_executor.parse_tool_request`.

        Args:
            text: A single line of LLM output to inspect.

        Returns:
            A dict with keys ``"tool"`` (str) and ``"args"`` (dict), or
            ``None`` when ``text`` does not contain a valid tool request.
        """
        return _parse_tool_request(text)

    def execute_tool(
        self,
        request: dict[str, Any],
        default_context: dict[str, Any],
        *,
        skip_policy_check: bool = False,
        skip_credential_check: bool = False,
        task_id: str | None = None,
        requested_by: str | None = None,
        skip_audit_log: bool = False,
    ) -> dict[str, Any]:
        """Execute a decoded tool request and return a result dictionary.

        Read-only compatibility calls may dispatch via OpenClaw HTTP when
        ``use_openclaw_tools`` is True and the gateway is configured. Mutations
        always use Rex's canonical executor/dispatcher instead of this direct path.

        Read-only HTTP behaviour:
        - ``200`` → returns the response dict from OpenClaw.
        - ``403`` → raises :class:`~rex.openclaw.tool_executor.PolicyDeniedError`.
        - ``404`` → tool not registered in OpenClaw; falls back to local.
        - ``429`` / ``5xx`` → follow :class:`~rex.openclaw.http_client.OpenClawClient`
          retry policy, then fall back locally when safe for the read operation.
        - Connection / auth errors → fall back to local execution.

        Args:
            request: Dict with ``"tool"`` and ``"args"`` keys.
            default_context: Ambient context (timezone, location, user, …).
            skip_policy_check: When *True*, bypass policy gating (local only).
            skip_credential_check: When *True*, bypass credential validation (local only).
            task_id: Optional correlation ID for audit logging (local only).
            requested_by: Optional identifier of the requesting entity (local only).
            skip_audit_log: When *True*, do not write an audit log entry (local only).

        Returns:
            A dict containing at minimum a ``"status"`` key (``"ok"`` or
            ``"error"``) and a ``"result"`` key with the tool output.

        Raises:
            PolicyDeniedError: If OpenClaw returns 403 for the tool call.
        """
        cfg = self._config
        if cfg is None:
            from rex.config import load_config as _load_config

            cfg = _load_config()

        client = get_openclaw_client(cfg)
        tool_name = request.get("tool", "")
        direct_operation = (
            _legacy_http_operation(tool_name) if isinstance(tool_name, str) else "mutation"
        )

        if cfg.use_openclaw_tools and client is not None and direct_operation == "read":
            args = request.get("args", {}) or {}
            payload: dict[str, Any] = {
                "tool": tool_name,
                "args": args,
                "sessionKey": default_context.get("session_key", "main"),
            }
            from rex.openclaw.capability_sync import get_openclaw_reconnect_controller

            reconnect = get_openclaw_reconnect_controller()

            def _dispatch_read() -> dict[str, Any]:
                return client.post("/tools/invoke", json=payload)

            def _disconnect_on_error(exc: Exception) -> bool:
                return (
                    isinstance(exc, (OpenClawConnectionError, OpenClawOutcomeUnknownError))
                    or isinstance(exc, OpenClawAPIError)
                    and exc.status >= 500
                )

            try:
                if reconnect is None:
                    result = _dispatch_read()
                else:
                    result = reconnect.dispatch_if_ready(
                        _dispatch_read,
                        disconnect_on_error=_disconnect_on_error,
                    )
                logger.debug("OpenClaw tool dispatch succeeded: tool=%s", tool_name)
                return result
            except OpenClawAPIError as exc:
                if exc.status == 403:
                    logger.warning("OpenClaw policy denied tool=%s: %s", tool_name, exc)
                    raise PolicyDeniedError(tool_name, str(exc)) from exc
                if exc.status == 404:
                    logger.info(
                        "Tool %s not found in OpenClaw (404), falling back to local",
                        tool_name,
                    )
                    # fall through to local execution below
                elif exc.status == 429 or exc.status >= 500:
                    _warn_gateway_fallback(
                        tool_name,
                        exc,
                        failure=f"OpenClawAPIError:{exc.status}",
                    )
                    # fall through to local execution below
                else:
                    raise
            except (
                OpenClawConnectionError,
                OpenClawAuthError,
                OpenClawOutcomeUnknownError,
                OpenClawUnavailableError,
            ) as exc:
                _warn_gateway_fallback(tool_name, exc)
                # fall through to local execution below

        return _execute_tool(
            request,
            default_context,
            skip_policy_check=skip_policy_check,
            skip_credential_check=skip_credential_check,
            task_id=task_id,
            requested_by=requested_by,
            skip_audit_log=skip_audit_log,
        )

    def route_if_tool_request(
        self,
        llm_text: str,
        default_context: dict[str, Any],
        model_call_fn: Callable[[dict[str, str]], str],
        *,
        skip_policy_check: bool = False,
    ) -> str:
        """Detect a tool call in *llm_text*, execute it, and return the final reply.

        If *llm_text* does not contain a tool request it is returned unchanged.

        Delegates to :func:`~rex.openclaw.tool_executor.route_if_tool_request`.

        Args:
            llm_text: Raw LLM output that may contain a tool-call line.
            default_context: Ambient context forwarded to :meth:`execute_tool`.
            model_call_fn: Callable that sends a follow-up message to the model
                and returns the model's response string.
            skip_policy_check: When *True*, bypass policy gating.

        Returns:
            The final text response (either *llm_text* verbatim, or the model's
            response after the tool result was injected).
        """
        return _route_if_tool_request(
            llm_text,
            default_context,
            model_call_fn,
            skip_policy_check=skip_policy_check,
        )

    # ------------------------------------------------------------------
    # Backward-compatible simple tool registration shim
    # ------------------------------------------------------------------

    def register_simple_tools(self) -> dict[str, Callable[..., dict[str, Any]]]:
        """Return built-in simple OpenClaw tool callables.

        Older tests/bootstrap paths expect a registration method that returns
        handles for basic tools. In the current architecture, tool dispatch is
        resolved dynamically by name via ``execute_tool`` and this explicit
        registration step is not required.
        """
        from rex.openclaw.tools.time_tool import time_now
        from rex.openclaw.tools.weather_tool import weather_now

        return {
            "time_now": time_now,
            "weather_now": weather_now,
        }
