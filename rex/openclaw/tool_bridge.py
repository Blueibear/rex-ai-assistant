"""OpenClaw tool bridge — US-P4-003 (updated US-P7-008, US-009).

Implements :class:`~rex.contracts.tool_routing.ToolRoutingProtocol` by
delegating to :mod:`rex.openclaw.tool_executor` module-level functions.

This bridge presents the ``ToolRoutingProtocol`` interface so that callers
do not need to import the internal tool executor directly.

When ``use_openclaw_tools`` is True and the OpenClaw gateway is configured,
:meth:`execute_tool` dispatches via HTTP POST to ``/tools/invoke``.  On a
404 response (tool not registered in OpenClaw) or connection failure, it
falls back to local execution transparently.  A 403 response raises
:class:`~rex.openclaw.tool_executor.PolicyDeniedError` so callers can
surface the denial to the user.

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
from typing import TYPE_CHECKING, Any, Callable

from rex.openclaw.errors import OpenClawAPIError, OpenClawAuthError, OpenClawConnectionError
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
from rex.openclaw.tools.business_tool import register_all_business_tools as _register_business_tools
from rex.openclaw.tools.calendar_tool import register as _register_calendar_create
from rex.openclaw.tools.email_tool import register as _register_send_email
from rex.openclaw.tools.ha_tool import register as _register_ha_call_service
from rex.openclaw.tools.plex_tool import register as _register_plex_tools
from rex.openclaw.tools.sms_tool import register as _register_send_sms
from rex.openclaw.tools.time_tool import register as _register_time_now
from rex.openclaw.tools.weather_tool import register as _register_weather_now
from rex.openclaw.tools.woocommerce_tool import register as _register_wc_tools
from rex.openclaw.tools.wordpress_tool import register as _register_wp_health_check

logger = logging.getLogger(__name__)


class ToolBridge:
    """Adapter that presents Rex's tool executor as an OpenClaw tool provider.

    Implements :class:`~rex.contracts.tool_routing.ToolRoutingProtocol` by
    delegating all three core operations to the corresponding module-level
    functions in :mod:`rex.openclaw.tool_executor`.

    When ``config.use_openclaw_tools`` is True and the gateway is reachable,
    :meth:`execute_tool` dispatches via HTTP instead of running locally.
    All other methods always run locally.

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

        Dispatches via OpenClaw HTTP when ``use_openclaw_tools`` is True and
        the gateway is configured; otherwise runs locally.

        HTTP behaviour:
        - ``200`` → returns the response dict from OpenClaw.
        - ``403`` → raises :class:`~rex.openclaw.tool_executor.PolicyDeniedError`.
        - ``404`` → tool not registered in OpenClaw; falls back to local.
        - ``429`` / ``5xx`` → retried by :class:`~rex.openclaw.http_client.OpenClawClient`;
          after retries exhausted raises :class:`~rex.openclaw.errors.OpenClawAPIError`.
        - Connection / auth errors → falls back to local execution.

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

        if cfg.use_openclaw_tools and client is not None:
            tool_name = request.get("tool", "")
            args = request.get("args", {}) or {}
            payload: dict[str, Any] = {
                "tool": tool_name,
                "args": args,
                "sessionKey": default_context.get("session_key", "main"),
            }
            try:
                result: dict[str, Any] = client.post("/tools/invoke", json=payload)
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
                else:
                    raise
            except (OpenClawConnectionError, OpenClawAuthError) as exc:
                logger.warning(
                    "OpenClaw tool dispatch error for tool=%s, falling back to local: %s",
                    tool_name,
                    exc,
                )
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
    # OpenClaw registration
    # ------------------------------------------------------------------

    def register_simple_tools(self, agent: Any = None) -> dict[str, Any]:
        """Register the simple read-only batch of tools with OpenClaw.

        Calls :func:`register` on each tool in the *simple / read-only* batch:

        * ``time_now`` — current local date and time
        * ``weather_now`` — current weather conditions

        Geolocation is handled internally by ``time_now`` / ``weather_now``
        via :mod:`rex.openclaw.tool_executor`'s fallback logic.  A standalone
        ``geolocation`` tool is not yet implemented in :mod:`rex.openclaw.tool_executor`
        and will be registered in a future iteration.

        Args:
            agent: Optional OpenClaw agent handle forwarded to each
                individual tool's :func:`register` call.

        Returns:
            A dict mapping tool name to the registration handle returned by
            each tool (``None`` when OpenClaw is not installed).
        """
        return {
            "time_now": _register_time_now(agent=agent),
            "weather_now": _register_weather_now(agent=agent),
        }

    def register_policy_gated_tools(self, agent: Any = None) -> dict[str, Any]:
        """Register the policy-gated tools batch with OpenClaw.

        Calls :func:`register` on each tool in the *policy-gated* batch:

        * ``send_email`` — send an email via Rex's EmailService (MEDIUM risk)
        * ``send_sms`` — send an SMS via Rex's SMSService (MEDIUM risk)
        * ``calendar_create`` — create a calendar event via Rex's CalendarService (MEDIUM risk)

        These tools require policy approval before execution in normal
        operation.  The approval flow is enforced by the PolicyAdapter
        (see :mod:`rex.openclaw.policy_adapter`), not by the tool callables
        themselves.

        Args:
            agent: Optional OpenClaw agent handle forwarded to each
                individual tool's :func:`register` call.

        Returns:
            A dict mapping tool name to the registration handle returned by
            each tool (``None`` when OpenClaw is not installed).
        """
        return {
            "send_email": _register_send_email(agent=agent),
            "send_sms": _register_send_sms(agent=agent),
            "calendar_create": _register_calendar_create(agent=agent),
        }

    def register_ha_tools(self, agent: Any = None) -> dict[str, Any]:
        """Register the Home Assistant tools batch with OpenClaw.

        Calls :func:`register` on the HA tool:

        * ``home_assistant_call_service`` — call a HA service (MEDIUM risk)

        This tool requires policy approval before execution in normal
        operation.  The approval flow is enforced by the PolicyAdapter
        (see :mod:`rex.openclaw.policy_adapter`), not by the tool callable
        itself.

        Args:
            agent: Optional OpenClaw agent handle forwarded to each
                individual tool's :func:`register` call.

        Returns:
            A dict mapping tool name to the registration handle returned by
            each tool (``None`` when OpenClaw is not installed).
        """
        return {
            "home_assistant_call_service": _register_ha_call_service(agent=agent),
        }

    def register_wordpress_tools(self, agent: Any = None) -> dict[str, Any]:
        """Register the WordPress tools batch with OpenClaw.

        Calls :func:`register` on the WordPress tool:

        * ``wordpress_health_check`` — read-only health check (LOW risk)

        WordPress integration is read-only; no write tools are registered.

        Args:
            agent: Optional OpenClaw agent handle forwarded to each
                individual tool's :func:`register` call.

        Returns:
            A dict mapping tool name to the registration handle returned by
            each tool (``None`` when OpenClaw is not installed).
        """
        return {
            "wordpress_health_check": _register_wp_health_check(agent=agent),
        }

    def register_woocommerce_tools(self, agent: Any = None) -> dict[str, Any]:
        """Register all WooCommerce tools batch with OpenClaw.

        Read tools (LOW risk):
        * ``wc_list_orders``      — list orders
        * ``wc_list_products``    — list products

        Write tools (HIGH risk, approval-gated):
        * ``wc_set_order_status`` — update order status
        * ``wc_create_coupon``    — create coupon
        * ``wc_disable_coupon``   — disable coupon

        Args:
            agent: Optional OpenClaw agent handle.

        Returns:
            A dict mapping tool name to the registration handle returned by
            each tool (``None`` when OpenClaw is not installed).
        """
        return _register_wc_tools(agent=agent)

    def register_business_tools(self, agent: Any = None) -> dict[str, Any]:
        """Register all business-domain tools (WooCommerce + WordPress) with OpenClaw.

        Convenience wrapper equivalent to calling both
        :meth:`register_woocommerce_tools` and :meth:`register_wordpress_tools`.

        Tools registered:

        * ``wc_list_orders`` — list WooCommerce orders (LOW risk)
        * ``wc_list_products`` — list WooCommerce products (LOW risk)
        * ``wc_set_order_status`` — update order status (HIGH risk, approval-gated)
        * ``wc_create_coupon`` — create coupon (HIGH risk, approval-gated)
        * ``wc_disable_coupon`` — disable coupon (HIGH risk, approval-gated)
        * ``wordpress_health_check`` — health check (LOW risk, read-only)

        Args:
            agent: Optional OpenClaw agent handle.

        Returns:
            A dict mapping every tool name to its registration handle
            (``None`` when OpenClaw is not installed).
        """
        return _register_business_tools(agent=agent)

    def register_plex_tools(self, agent: Any = None) -> dict[str, Any]:
        """Register all Plex tools batch with OpenClaw.

        Tools (LOW risk, read-only search + playback control):
        * ``plex_search`` — search the Plex library
        * ``plex_play``   — start playback on a Plex client
        * ``plex_pause``  — pause playback on a Plex client
        * ``plex_stop``   — stop playback on a Plex client

        Args:
            agent: Optional OpenClaw agent handle.

        Returns:
            A dict mapping tool name to the registration handle returned by
            each tool (``None`` when OpenClaw is not installed).
        """
        return _register_plex_tools(agent=agent)

    def register(self, agent: Any = None) -> Any:
        """Register this bridge as the OpenClaw tool provider.

        When ``openclaw`` is installed, this method registers the bridge so
        that OpenClaw routes tool calls through Rex's tool router.  When
        OpenClaw is absent, logs a warning and returns ``None``.

        .. note::
            The exact OpenClaw tool-provider registration call is a stub (see
            PRD §8.3 — *"Confirm OpenClaw's tool registration mechanism"*).
            Replace the ``# TODO`` below once the API is confirmed.

        Args:
            agent: Optional OpenClaw agent handle.

        Returns:
            The registration handle from OpenClaw, or ``None``.
        """
        from rex.config import load_config as _load_config
        from rex.openclaw.http_client import get_openclaw_client

        if get_openclaw_client(_load_config()) is None:
            logger.warning(
                "OpenClaw gateway not configured — ToolBridge not registered as tool provider"
            )
            return None

        # TODO: replace with real OpenClaw tool provider registration once API is confirmed.
        # Expected shape (to be verified):
        #   handle = _openclaw.register_tool_provider(
        #       provider=self,
        #       agent=agent,
        #   )
        #   return handle
        logger.warning(
            "OpenClaw tool provider registration stub — update once API is confirmed (PRD §8.3)"
        )
        return None
