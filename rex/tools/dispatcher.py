"""Auto tool selection — map user intent to tools (US-TD-002/US-TD-003).

``ToolDispatcher.select_tools()`` maps a user message to zero or more
registered tools via keyword/intent matching.  ``execute_tools()`` invokes
each selected tool with a configurable timeout and a single retry on
transient network errors.

Canonical implementation of ``ToolDispatcherProtocol`` (see
``rex.tools.protocol``).  The ``dispatch(name, args, context)`` signature
defined in the protocol will be added to this class in a follow-up story;
the current ``execute_tools()`` API is preserved for backward compatibility.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from .execution import _is_auth_error as _is_auth_error
from .execution import _is_transient_error as _is_transient_error
from .registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent rule table
# Each entry: (capability_tag, compiled_pattern)
# The tag must match at least one of a tool's capability_tags.
# ---------------------------------------------------------------------------

_INTENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "email",
        re.compile(
            r"\b(email|mail|inbox|send\s+an?\s+email|read\s+my\s+email|"
            r"check\s+my\s+(email|mail)|new\s+message[s]?|unread|compose)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "weather",
        re.compile(
            r"\b(weather|forecast|temperature|rain|snow|sunny|cloudy|"
            r"humidity|wind|storm|outside|degrees?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "search",
        re.compile(
            r"\b(search|look\s+up|find\s+out|google|browse|web|news|"
            r"latest|what\s+is\s+the\s+latest|who\s+is|tell\s+me\s+about)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "calendar",
        re.compile(
            r"\b(calendar|schedule|appointment|meeting|event|remind\s+me|"
            r"agenda|book\s+(a|an|the)\s+\w+|add\s+(to|an?)\s+(my\s+)?calendar)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "smart_home",
        re.compile(
            r"\b(turn\s+(on|off)|lights?|thermostat|lock|unlock|home\s+assistant|"
            r"smart\s+home|dim|brighten|set\s+the\s+(lights?|temperature|thermostat)|"
            r"open\s+(the\s+)?(garage|door)|close\s+(the\s+)?(garage|door))\b",
            re.IGNORECASE,
        ),
    ),
]

# ---------------------------------------------------------------------------
# Error classification helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ToolDispatcher
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT: float = 10.0


class ToolDispatcher:
    """Select and execute tools from a registry.

    Args:
        registry: The ``ToolRegistry`` to pull tools from.
        config: Optional ``AppConfig`` instance.  Used to filter unavailable
            tools and to read ``tool_timeout_seconds``.  When *None* all
            registered tools are candidates and the default timeout applies.

    Usage::

        dispatcher = ToolDispatcher(registry, config=app_config)
        tools = dispatcher.select_tools("What's the weather and check my email?")
        results = dispatcher.execute_tools(tools, message)
        context = dispatcher.format_tool_context(results)
    """

    def __init__(
        self,
        registry: ToolRegistry,
        config: Any = None,
    ) -> None:
        self._registry = registry
        self._config = config
        self._timeout_seconds: float = float(
            getattr(config, "tool_timeout_seconds", _DEFAULT_TIMEOUT) or _DEFAULT_TIMEOUT
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_tools(
        self,
        message: str,
        *,
        user_id: str | None = None,
        granted_permissions: set[str] | frozenset[str] | None = None,
    ) -> list[Tool]:
        """Return authorized tools whose domain matches the user's intent in *message*.

        Intent detection is keyword-based.  Each candidate tool is scored by
        the number of its ``capability_tags`` that appear in the set of tags
        triggered by matching intent rules.  Tools are returned sorted by
        confidence (score) descending so that the highest-confidence match
        comes first.  Multiple tools are returned when the message spans
        multiple domains (e.g. "weather and email").  Returns an empty list
        when no intent is matched — the caller falls back to the LLM path.

        Args:
            message: The raw user transcript or chat message.

        Returns:
            Confidence-sorted list of matched ``Tool`` objects (deduped).
        """
        current_permissions = granted_permissions
        if current_permissions is None and user_id:
            try:
                from rex.permissions import get_permissions  # noqa: PLC0415

                current_permissions = frozenset(get_permissions(user_id))
            except Exception:
                logger.exception(
                    "tool_dispatcher: failed to resolve permissions for user %r", user_id
                )
                current_permissions = frozenset()

        if self._config is not None:
            candidates = self._registry.available_tools(
                self._config, granted_permissions=current_permissions
            )
        else:
            candidates = self._registry.all_tools()
            if current_permissions is not None:
                candidates = [
                    tool
                    for tool in candidates
                    if self._registry.capability_registry.is_authorized(
                        tool.name, current_permissions
                    )
                ]

        # Determine which capability tags are triggered by matching intent rules.
        fired_tags: set[str] = set()
        for capability_tag, pattern in _INTENT_RULES:
            if pattern.search(message):
                fired_tags.add(capability_tag)
                logger.debug("tool_dispatcher: intent rule %r fired", capability_tag)

        if not fired_tags:
            logger.debug("tool_dispatcher: no intent match for message")
            return []

        # Score each candidate: count of capability_tags that appear in fired_tags.
        seen_names: set[str] = set()
        scored: list[tuple[int, Tool]] = []
        for tool in candidates:
            score = sum(1 for tag in tool.capability_tags if tag in fired_tags)
            if score > 0 and tool.name not in seen_names:
                seen_names.add(tool.name)
                scored.append((score, tool))
                logger.debug(
                    "tool_dispatcher: tool=%r score=%d",
                    tool.name,
                    score,
                )

        # Sort by confidence descending; stable sort preserves registration
        # order for tools with equal scores.
        scored.sort(key=lambda x: x[0], reverse=True)
        return [tool for _, tool in scored]

    def select_tools_for_user(
        self,
        message: str,
        *,
        user_id: str,
        granted_permissions: set[str] | frozenset[str] | None = None,
    ) -> list[Tool]:
        """Select tools with the current user's live authorization context."""
        return self.select_tools(
            message,
            user_id=user_id,
            granted_permissions=granted_permissions,
        )

    def execute_tools(
        self, tools: list[Tool], message: str, *, user_id: str | None = None
    ) -> dict[str, Any]:
        """Invoke *tools* with timeout + one-retry on transient errors.

        Each handler is called with ``transcript=message``.  For each tool:

        * If the call succeeds within the timeout the result is stored.
        * If the call times out the message
          ``"I couldn't reach {name} in time"`` is stored.
        * If the call raises a transient error (network, HTTP 5xx) it is
          retried **once**.  Auth errors are never retried.
        * All invocations are logged with tool name, duration, and
          success/failure.

        Args:
            tools:   Tools to execute (from :meth:`select_tools`).
            message: The user message passed as ``transcript`` kwarg.
            user_id: Active user identifier forwarded to each tool handler as
                     ``_user_id`` so that user-scoped tools (e.g. email) can
                     enforce per-user access control.

        Returns:
            Dict mapping tool name to its result (or error/timeout string).
        """
        results: dict[str, Any] = {}
        for tool in tools:
            start = time.monotonic()
            result = self.dispatch(
                tool.name,
                {"transcript": message},
                {"user_id": user_id} if user_id is not None else {},
            )
            duration = time.monotonic() - start
            logger.info(
                "tool_dispatcher: %r %.3fs %s",
                tool.name,
                duration,
                "ok" if result.success else result.status,
            )
            if result.success:
                results[tool.name] = result.output
            elif result.error == "Execution timed out":
                results[tool.name] = f"I couldn't reach {tool.name} in time"
            else:
                results[tool.name] = (
                    f"[tool error: {result.detail or result.error or 'unknown error'}]"
                )
        return results

    def dispatch(
        self,
        name: str,
        args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Implements ``ToolDispatcherProtocol.dispatch()``.

        Looks up *name* in the registry and invokes its handler with *args*
        as keyword arguments.  Returns a ``ToolResult`` (from
        :mod:`rex.tools.protocol`).

        Args:
            name:    Tool name to invoke.
            args:    Keyword arguments forwarded to the handler.
            context: Optional execution context (not forwarded to handler;
                     reserved for future middleware use).

        Returns:
            ``ToolResult(success=True, output=...)`` on success or
            ``ToolResult(success=False, error=...)`` on failure / timeout.
        """
        from rex.tools.execution import ToolExecutionLifecycle
        from rex.tools.protocol import ToolResult  # local import — avoids circular dependency

        tool = self._registry.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name!r}")

        from rex.mobile_api.action_context import authorized_mobile_tool  # noqa: PLC0415

        available = self._config is None or tool in self._registry.available_tools(self._config)
        with authorized_mobile_tool(
            tool.name,
            capability_tags=tool.capability_tags,
            operation=getattr(tool, "operation", None),
            arguments=args,
        ):
            return ToolExecutionLifecycle().execute(
                tool,
                args,
                context,
                timeout_seconds=self._timeout_seconds,
                available=available,
            )

    @staticmethod
    def format_tool_context(results: dict[str, Any]) -> str:
        """Format *results* dict as a context block for the LLM prompt.

        Returns an empty string when *results* is empty.
        """
        if not results:
            return ""
        lines = ["[Tool results:"]
        for name, value in results.items():
            lines.append(f"  {name}: {value}")
        lines.append("]")
        return "\n".join(lines)
