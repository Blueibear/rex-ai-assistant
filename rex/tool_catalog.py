"""Authoritative catalog of tools that are executable end-to-end.

This module defines the single source of truth for which tools have real,
working handlers.  The Planner, ToolRegistry, and ToolRouter all validate
against this set so that the execution surface stays consistent.

Adding a tool name here is a commitment that a real handler exists in
rex/tool_router.py that can invoke it without raising NotImplementedError.
"""

from __future__ import annotations

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

__all__ = ["EXECUTABLE_TOOLS"]
