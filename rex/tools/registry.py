"""Tool registry for Rex auto-dispatch (Phase 5 — US-TD-001).

Catalogs every tool Rex can invoke with metadata for automatic selection.
``available_tools()`` filters by which AppConfig fields are satisfied so
only truly callable tools are offered to the dispatcher.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Metadata and handler for a single Rex tool.

    Attributes:
        name: Unique tool identifier (e.g. ``"web_search"``).
        description: Human-readable description of what the tool does.
        capability_tags: Category labels used for intent matching (e.g.
            ``["search", "web"]``).
        requires_config: ``AppConfig`` attribute names that must be truthy
            for this tool to be available.  An empty list means always
            available.
        handler: Callable invoked when the tool is dispatched.  Signature
            is tool-specific; the dispatcher passes keyword arguments.
    """

    name: str
    description: str
    capability_tags: list[str]
    requires_config: list[str]
    handler: Callable[..., Any]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Tool name cannot be empty")
        if not self.description:
            raise ValueError("Tool description cannot be empty")


class ToolRegistry:
    """Registry of all Rex tools with config-based availability filtering.

    Usage::

        registry = ToolRegistry()
        registry.register(Tool(...))
        available = registry.available_tools(app_config)
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, tool: Tool) -> None:
        """Register *tool*, replacing any existing entry with the same name."""
        if tool.name in self._tools:
            logger.debug("tool_registry: replacing existing tool %r", tool.name)
        self._tools[tool.name] = tool
        logger.debug("tool_registry: registered %r", tool.name)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, name: str) -> Tool | None:
        """Return the tool with *name* or ``None`` if not registered."""
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        """Return all registered tools regardless of availability."""
        return list(self._tools.values())

    def available_tools(self, config: Any) -> list[Tool]:
        """Return tools whose ``requires_config`` fields are satisfied.

        A field is *satisfied* when ``getattr(config, field_name, None)``
        is truthy.

        Args:
            config: An ``AppConfig`` instance (or any object with the
                expected attributes).

        Returns:
            Subset of registered tools that are fully configured.
        """
        result: list[Tool] = []
        for tool in self._tools.values():
            if self._is_available(tool, config):
                result.append(tool)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_available(self, tool: Tool, config: Any) -> bool:
        for attr in tool.requires_config:
            val = getattr(config, attr, None)
            if not val:
                logger.debug(
                    "tool_registry: %r unavailable — config.%s not set",
                    tool.name,
                    attr,
                )
                return False
        return True


# ---------------------------------------------------------------------------
# Default registry populated with all built-in Rex tools
# ---------------------------------------------------------------------------


def _noop_handler(**kwargs: Any) -> dict[str, Any]:
    """Placeholder handler for tools that delegate to another executor at runtime."""
    return {}


def _build_default_registry() -> ToolRegistry:
    """Build and return a ``ToolRegistry`` pre-populated with all Rex tools."""
    # Lazily import so optional dependencies don't block startup.
    from rex.openclaw.tools.calendar_tool import calendar_create
    from rex.openclaw.tools.email_tool import send_email
    from rex.openclaw.tools.ha_tool import ha_call_service
    from rex.openclaw.tools.sms_tool import send_sms
    from rex.openclaw.tools.time_tool import time_now as _time_now
    from rex.openclaw.tools.weather_tool import weather_now
    from rex.tools.file_ops import read_file as _read_file
    from rex.tools.windows_diagnostics import (
        get_battery_status,
        get_cpu_usage,
        get_disk_usage,
        get_memory_usage,
        get_system_info,
        list_processes,
    )

    registry = ToolRegistry()

    registry.register(
        Tool(
            name="time_now",
            description="Get the current local date and time for a given location.",
            capability_tags=["time", "clock", "date"],
            requires_config=[],
            handler=_time_now,
        )
    )

    registry.register(
        Tool(
            name="weather_now",
            description="Get the current weather conditions for a given location.",
            capability_tags=["weather", "forecast"],
            requires_config=["openweathermap_api_key"],
            handler=weather_now,
        )
    )

    registry.register(
        Tool(
            name="web_search",
            description=(
                "Search the web for up-to-date information using configured "
                "search providers (Brave, SerpAPI, DuckDuckGo, Google CSE)."
            ),
            capability_tags=["search", "web", "lookup"],
            requires_config=["brave_api_key"],
            handler=_noop_handler,
        )
    )

    registry.register(
        Tool(
            name="send_email",
            description="Send an email to one or more recipients.",
            capability_tags=["email", "messaging", "send"],
            requires_config=["email_accounts"],
            handler=send_email,
        )
    )

    registry.register(
        Tool(
            name="calendar_create",
            description="Create a new calendar event.",
            capability_tags=["calendar", "schedule", "event"],
            requires_config=[],
            handler=calendar_create,
        )
    )

    registry.register(
        Tool(
            name="home_assistant_call_service",
            description="Control smart home devices via Home Assistant.",
            capability_tags=["smart_home", "home_assistant", "iot"],
            requires_config=["ha_base_url", "ha_token"],
            handler=ha_call_service,
        )
    )

    registry.register(
        Tool(
            name="send_sms",
            description="Send an SMS text message to a phone number.",
            capability_tags=["sms", "messaging", "text"],
            requires_config=[],
            handler=send_sms,
        )
    )

    registry.register(
        Tool(
            name="file_ops",
            description=(
                "Read, write, list, or move files on the local filesystem "
                "within allowed directories."
            ),
            capability_tags=["file", "filesystem", "local"],
            requires_config=[],
            handler=_read_file,
        )
    )

    registry.register(
        Tool(
            name="get_system_info",
            description="Get OS and hardware information (platform, CPU count, total RAM, boot time).",
            capability_tags=["windows", "diagnostics", "system"],
            requires_config=[],
            handler=get_system_info,
        )
    )

    registry.register(
        Tool(
            name="get_cpu_usage",
            description="Get current CPU usage percentage and frequency.",
            capability_tags=["windows", "diagnostics", "cpu"],
            requires_config=[],
            handler=get_cpu_usage,
        )
    )

    registry.register(
        Tool(
            name="get_memory_usage",
            description="Get current RAM and swap memory usage statistics.",
            capability_tags=["windows", "diagnostics", "memory", "ram"],
            requires_config=[],
            handler=get_memory_usage,
        )
    )

    registry.register(
        Tool(
            name="get_disk_usage",
            description="Get disk usage for all mounted partitions.",
            capability_tags=["windows", "diagnostics", "disk", "storage"],
            requires_config=[],
            handler=get_disk_usage,
        )
    )

    registry.register(
        Tool(
            name="get_battery_status",
            description="Get battery charge level and charging status.",
            capability_tags=["windows", "diagnostics", "battery"],
            requires_config=[],
            handler=get_battery_status,
        )
    )

    registry.register(
        Tool(
            name="list_processes",
            description="List running processes sorted by CPU usage.",
            capability_tags=["windows", "diagnostics", "processes"],
            requires_config=[],
            handler=list_processes,
        )
    )

    return registry


#: Module-level singleton — lazily initialised on first call.
_default_registry: ToolRegistry | None = None


def get_default_registry() -> ToolRegistry:
    """Return the module-level default ``ToolRegistry`` (lazy init)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = _build_default_registry()
    return _default_registry
