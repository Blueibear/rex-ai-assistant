"""Authoritative catalog of tools that are executable end-to-end.

This module defines the single source of truth for which tools have real,
working handlers.  The Planner, ToolRegistry, and ToolRouter all validate
against this set so that the execution surface stays consistent.

Adding a tool name here is a commitment that a real handler exists in
rex/tool_router.py that can invoke it without raising NotImplementedError.

``TOOL_CATALOG`` is the authoritative registry of tools with their intent
patterns for auto-selection.  ``ToolDispatcher`` derives its intent rules
from this catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Tools with real end-to-end handlers.
#:
#: ==================  ==========================================
#: Tool name           Handler location
#: ==================  ==========================================
#: time_now            rex/tool_router.py – returns local/UTC time
#: weather_now         rex/tool_router.py – calls weather provider
#: web_search          rex/tool_router.py – calls search provider
#: send_email          rex/tool_router.py – calls EmailService.send()
#: calendar_create_event  rex/tool_router.py – calls CalendarService
#: home_assistant_call_service  rex/tool_router.py – calls HA HTTP API
#: ==================  ==========================================
EXECUTABLE_TOOLS: frozenset[str] = frozenset(
    {
        "time_now",
        "weather_now",
        "web_search",
        "send_email",
        "calendar_create_event",
        "home_assistant_call_service",
    }
)


@dataclass
class CatalogEntry:
    """Registry entry mapping a tool to its intent patterns.

    Attributes:
        name: Unique tool identifier (must match the ``Tool.name`` in the
            ``ToolRegistry``).
        description: Human-readable description of what the tool does.
        intent_patterns: List of regex pattern strings used for intent
            detection.  A user message is routed to this tool when any
            pattern matches.
        capability_tags: Category labels used for scoring and deduplication
            (must align with ``Tool.capability_tags`` in the registry).
        requires_config: ``AppConfig`` attribute names that must be truthy
            for this tool to be active.
    """

    name: str
    description: str
    intent_patterns: list[str]
    capability_tags: list[str] = field(default_factory=list)
    requires_config: list[str] = field(default_factory=list)


#: Catalog of all tools with their intent patterns for auto-selection.
TOOL_CATALOG: list[CatalogEntry] = [
    CatalogEntry(
        name="weather_now",
        description="Get current weather conditions for a location.",
        intent_patterns=[
            r"\b(weather|forecast|temperature|rain|snow|sunny|cloudy|"
            r"humidity|wind|storm|outside|degrees?)\b",
        ],
        capability_tags=["weather", "forecast"],
        requires_config=["openweathermap_api_key"],
    ),
    CatalogEntry(
        name="web_search",
        description="Search the web for up-to-date information.",
        intent_patterns=[
            r"\b(search|look\s+up|find\s+out|google|browse|web|news|"
            r"latest|what\s+is\s+the\s+latest|who\s+is|tell\s+me\s+about)\b",
        ],
        capability_tags=["search", "web", "lookup"],
        requires_config=["brave_api_key"],
    ),
    CatalogEntry(
        name="home_assistant_call_service",
        description="Control smart home devices via Home Assistant.",
        intent_patterns=[
            r"\b(turn\s+(on|off)|lights?|thermostat|lock|unlock|"
            r"home\s+assistant|smart\s+home|dim|brighten|"
            r"set\s+the\s+(lights?|temperature|thermostat)|"
            r"open\s+(the\s+)?(garage|door)|close\s+(the\s+)?(garage|door))\b",
        ],
        capability_tags=["smart_home", "home_assistant", "iot"],
        requires_config=["ha_base_url", "ha_token"],
    ),
    CatalogEntry(
        name="calendar_create_event",
        description="Create or query calendar events.",
        intent_patterns=[
            r"\b(calendar|schedule|appointment|meeting|event|remind\s+me|"
            r"agenda|book\s+(a|an|the)\s+\w+|add\s+(to|an?)\s+(my\s+)?calendar)\b",
        ],
        capability_tags=["calendar", "schedule", "event"],
    ),
    CatalogEntry(
        name="send_email",
        description="Send or check email.",
        intent_patterns=[
            r"\b(email|mail|inbox|send\s+an?\s+email|read\s+my\s+email|"
            r"check\s+my\s+(email|mail)|new\s+message[s]?|unread|compose)\b",
        ],
        capability_tags=["email", "messaging", "send"],
        requires_config=["email_accounts"],
    ),
    CatalogEntry(
        name="time_now",
        description="Get the current local date and time.",
        intent_patterns=[
            r"\b(time|clock|what\s+time|current\s+time|date\s+today|"
            r"today('?s)?\s+date|what\s+day)\b",
        ],
        capability_tags=["time", "clock", "date"],
    ),
]


__all__ = ["CatalogEntry", "EXECUTABLE_TOOLS", "TOOL_CATALOG"]
