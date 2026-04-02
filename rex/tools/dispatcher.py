"""Auto tool selection — map user intent to tools (US-TD-002/US-TD-003).

``ToolDispatcher.select_tools()`` maps a user message to zero or more
registered tools via keyword/intent matching.  ``execute_tools()`` invokes
each selected tool with a configurable timeout and a single retry on
transient network errors.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import time
from typing import Any

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

#: Exception types that indicate a transient network problem (retry once).
_TRANSIENT_TYPES = (
    TimeoutError,
    ConnectionError,
    ConnectionResetError,
    ConnectionRefusedError,
    ConnectionAbortedError,
    OSError,
)

#: Exception types that indicate an auth failure (never retry).
_AUTH_TYPES = (PermissionError,)


def _is_transient_error(exc: BaseException) -> bool:
    """Return True if *exc* is a retriable network/transient error.

    Also recognises HTTP-style errors with a ``status_code`` attribute
    whose value is >= 500 (server-side transient).
    """
    if isinstance(exc, _TRANSIENT_TYPES):
        return True
    # HTTP-style exception with a status code
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and status >= 500:
        return True
    # Explicit opt-in marker on custom exceptions
    return bool(getattr(exc, "is_transient", False))


def _is_auth_error(exc: BaseException) -> bool:
    """Return True if *exc* is an authentication / authorisation failure."""
    if isinstance(exc, _AUTH_TYPES):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and status in (401, 403):
        return True
    return bool(getattr(exc, "auth_error", False))


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

    def select_tools(self, message: str) -> list[Tool]:
        """Return tools whose domain matches the user's intent in *message*.

        Intent detection is keyword-based; multiple tools are returned when
        the message spans multiple domains.  Returns an empty list when no
        intent is matched (normal LLM path).

        Args:
            message: The raw user transcript or chat message.

        Returns:
            Ordered list of matched ``Tool`` objects (deduped, stable order).
        """
        if self._config is not None:
            candidates = self._registry.available_tools(self._config)
        else:
            candidates = self._registry.all_tools()

        # Build a lookup: capability_tag → list[Tool]
        tag_index: dict[str, list[Tool]] = {}
        for tool in candidates:
            for tag in tool.capability_tags:
                tag_index.setdefault(tag, []).append(tool)

        selected_names: set[str] = set()
        selected: list[Tool] = []

        for capability_tag, pattern in _INTENT_RULES:
            if pattern.search(message):
                matched_tools = tag_index.get(capability_tag, [])
                for tool in matched_tools:
                    if tool.name not in selected_names:
                        selected_names.add(tool.name)
                        selected.append(tool)
                        logger.debug(
                            "tool_dispatcher: intent=%r matched tool=%r",
                            capability_tag,
                            tool.name,
                        )

        if not selected:
            logger.debug("tool_dispatcher: no intent match for message")

        return selected

    def execute_tools(self, tools: list[Tool], message: str) -> dict[str, Any]:
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
            tools: Tools to execute (from :meth:`select_tools`).
            message: The user message passed as ``transcript`` kwarg.

        Returns:
            Dict mapping tool name to its result (or error/timeout string).
        """
        results: dict[str, Any] = {}
        for tool in tools:
            start = time.monotonic()
            value, ok = self._invoke_with_timeout_retry(tool, message)
            duration = time.monotonic() - start
            logger.info(
                "tool_dispatcher: %r %.3fs %s",
                tool.name,
                duration,
                "ok" if ok else "failed",
            )
            results[tool.name] = value
        return results

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invoke_with_timeout_retry(self, tool: Tool, message: str) -> tuple[Any, bool]:
        """Invoke *tool* handler; retry once on transient errors.

        Returns:
            ``(result_value, success_bool)``
        """
        timeout = self._timeout_seconds
        last_exc: BaseException | None = None

        for attempt in range(2):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(tool.handler, transcript=message)
                try:
                    result = future.result(timeout=timeout)
                    return result, True
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    logger.warning(
                        "tool_dispatcher: %r timed out after %.1fs",
                        tool.name,
                        timeout,
                    )
                    return f"I couldn't reach {tool.name} in time", False
                except Exception as exc:
                    last_exc = exc
                    if attempt == 0 and _is_transient_error(exc) and not _is_auth_error(exc):
                        logger.debug(
                            "tool_dispatcher: %r transient error on attempt 1, retrying: %s",
                            tool.name,
                            exc,
                        )
                        continue
                    # Non-transient or second attempt — fail fast
                    break

        exc_msg = str(last_exc) if last_exc is not None else "unknown error"
        logger.warning("tool_dispatcher: %r failed: %s", tool.name, exc_msg)
        return f"[tool error: {exc_msg}]", False
