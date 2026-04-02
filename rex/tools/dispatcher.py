"""Auto tool selection — map user intent to tools (US-TD-002).

``ToolDispatcher.select_tools()`` maps a user message to zero or more
registered tools via keyword/intent matching.  When multiple domains are
detected all matching tools are returned so the caller can invoke them in
parallel and aggregate results as LLM context.
"""

from __future__ import annotations

import logging
import re
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


class ToolDispatcher:
    """Select tools from a registry based on keyword/intent matching.

    Args:
        registry: The ``ToolRegistry`` to pull tools from.
        config: Optional ``AppConfig`` instance used to filter unavailable
            tools.  When *None* all registered tools are candidates.

    Usage::

        dispatcher = ToolDispatcher(registry, config=app_config)
        tools = dispatcher.select_tools("What's the weather and check my email?")
        # returns [Tool(weather_now), Tool(send_email)]
    """

    def __init__(
        self,
        registry: ToolRegistry,
        config: Any = None,
    ) -> None:
        self._registry = registry
        self._config = config

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
        """Invoke *tools* and return a mapping of tool name → result.

        Each handler is called with ``transcript=message``.  Exceptions are
        caught and stored as strings so one failing tool doesn't block others.

        Args:
            tools: Tools to execute (from :meth:`select_tools`).
            message: The user message passed as ``transcript`` kwarg.

        Returns:
            Dict mapping tool name to its result (or error string).
        """
        results: dict[str, Any] = {}
        for tool in tools:
            try:
                result = tool.handler(transcript=message)
                results[tool.name] = result
                logger.info("tool_dispatcher: executed %r successfully", tool.name)
            except Exception as exc:
                logger.warning("tool_dispatcher: %r raised %s", tool.name, exc)
                results[tool.name] = f"[tool error: {exc}]"
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
