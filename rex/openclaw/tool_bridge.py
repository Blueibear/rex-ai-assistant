"""OpenClaw tool bridge — US-P4-003.

Implements :class:`~rex.contracts.tool_routing.ToolRoutingProtocol` by
delegating to Rex's existing ``rex.tool_router`` module-level functions.

This bridge is the first step in routing tool calls through OpenClaw.  It
presents the ``ToolRoutingProtocol`` interface so that callers do not need
to import ``rex.tool_router`` directly and can be swapped once the full
OpenClaw tool-dispatch API is confirmed (PRD §8.3).

When the ``openclaw`` package is not installed, :func:`register` logs a
warning and returns ``None``.  All other methods work without OpenClaw
installed because they delegate to the existing Rex tool router.

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
from importlib.util import find_spec
from typing import Any, Callable

from rex.openclaw.tools.calendar_tool import register as _register_calendar_create
from rex.openclaw.tools.email_tool import register as _register_send_email
from rex.openclaw.tools.sms_tool import register as _register_send_sms
from rex.openclaw.tools.time_tool import register as _register_time_now
from rex.openclaw.tools.weather_tool import register as _register_weather_now
from rex.tool_router import (
    execute_tool as _execute_tool,
    parse_tool_request as _parse_tool_request,
    route_if_tool_request as _route_if_tool_request,
)

logger = logging.getLogger(__name__)

OPENCLAW_AVAILABLE: bool = find_spec("openclaw") is not None

if OPENCLAW_AVAILABLE:  # pragma: no cover
    import openclaw as _openclaw  # type: ignore[import-not-found]
else:
    _openclaw = None  # type: ignore[assignment]


class ToolBridge:
    """Adapter that presents Rex's tool router as an OpenClaw tool provider.

    Implements :class:`~rex.contracts.tool_routing.ToolRoutingProtocol` by
    delegating all three core operations to the corresponding module-level
    functions in :mod:`rex.tool_router`.

    When ``openclaw`` is installed, :meth:`register` registers the bridge
    as the tool provider so that OpenClaw dispatches tool calls through Rex
    (stub — filled in once the OpenClaw tool-provider API is confirmed, see
    PRD §8.3).

    Without OpenClaw the bridge is still useful as a thin convenience
    wrapper that satisfies the :class:`~rex.contracts.tool_routing.ToolRoutingProtocol`
    structural type.
    """

    # ------------------------------------------------------------------
    # ToolRoutingProtocol implementation
    # ------------------------------------------------------------------

    def parse_tool_request(self, text: str) -> dict[str, Any] | None:
        """Return a parsed tool-request dict or *None* if not a tool call.

        Delegates to :func:`~rex.tool_router.parse_tool_request`.

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

        Delegates to :func:`~rex.tool_router.execute_tool`.

        Args:
            request: Dict with ``"tool"`` and ``"args"`` keys.
            default_context: Ambient context (timezone, location, user, …).
            skip_policy_check: When *True*, bypass policy gating.
            skip_credential_check: When *True*, bypass credential validation.
            task_id: Optional correlation ID for audit logging.
            requested_by: Optional identifier of the requesting entity.
            skip_audit_log: When *True*, do not write an audit log entry.

        Returns:
            A dict containing at minimum a ``"status"`` key (``"ok"`` or
            ``"error"``) and a ``"result"`` key with the tool output.
        """
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

        Delegates to :func:`~rex.tool_router.route_if_tool_request`.

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
        via :mod:`rex.tool_router`'s fallback logic.  A standalone
        ``geolocation`` tool is not yet implemented in :mod:`rex.tool_router`
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
        if not OPENCLAW_AVAILABLE:
            logger.warning(
                "openclaw package not installed — ToolBridge not registered as tool provider"
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
