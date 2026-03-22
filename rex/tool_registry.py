# OPENCLAW-REPLACE: This module will be replaced by OpenClaw. Do not add new features.

"""Tool registry for managing tool metadata and health checks.

This module provides a centralized registry for tools that:
- Stores metadata about available tools (name, description, capabilities)
- Tracks required credentials for each tool
- Provides health check functionality to verify tool availability
- Allows tools to self-register without modifying core code

Tools query the CredentialManager when invoked, not at registration time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from rex.credentials import CredentialManager, get_credential_manager

logger = logging.getLogger(__name__)

# Type alias for health check function
HealthCheckFn = Callable[[], tuple[bool, str]]


def _default_health_check() -> tuple[bool, str]:
    """Default health check that always passes."""
    return True, "OK"


@dataclass
class ToolMeta:
    """Metadata about a registered tool.

    Attributes:
        name: Unique identifier for the tool (e.g., "time_now").
        description: Human-readable description of what the tool does.
        required_credentials: List of service names needed (e.g., ["email", "calendar"]).
        capabilities: List of capability tags (e.g., ["read", "write", "network"]).
        health_check: Callable that returns (ok: bool, message: str).
        version: Optional version string for the tool.
        enabled: Whether the tool is currently enabled.
    """

    name: str
    description: str
    required_credentials: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    health_check: HealthCheckFn = field(default=_default_health_check)
    version: str = "1.0.0"
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate tool metadata after initialization."""
        if not self.name:
            raise ValueError("Tool name cannot be empty")
        if not self.description:
            raise ValueError("Tool description cannot be empty")


class ToolNotFoundError(Exception):
    """Raised when a requested tool is not found in the registry."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' not found in registry")


class MissingCredentialError(Exception):
    """Raised when required credentials are not available for a tool."""

    def __init__(self, tool_name: str, missing_credentials: list[str]) -> None:
        self.tool_name = tool_name
        self.missing_credentials = missing_credentials
        creds = ", ".join(missing_credentials)
        super().__init__(
            f"Tool '{tool_name}' requires credentials that are not configured: {creds}"
        )


class ToolRegistry:
    """Central registry for tool metadata and health checks.

    The ToolRegistry maintains a collection of registered tools and provides
    methods to:
    - Register new tools with their metadata
    - Look up tools by name
    - Check if tools have required credentials available
    - Run health checks on individual tools or all tools

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register_tool(ToolMeta(
        ...     name="weather",
        ...     description="Get current weather",
        ...     required_credentials=["weather_api"],
        ... ))
        >>> tool = registry.get_tool("weather")
        >>> print(tool.description)
        Get current weather
    """

    def __init__(
        self,
        *,
        credential_manager: CredentialManager | None = None,
    ) -> None:
        """Initialize the tool registry.

        Args:
            credential_manager: Optional CredentialManager instance.
                If not provided, uses the global instance.
        """
        self._tools: dict[str, ToolMeta] = {}
        self._credential_manager = credential_manager

    @property
    def credential_manager(self) -> CredentialManager:
        """Get the credential manager (lazy initialization)."""
        if self._credential_manager is None:
            self._credential_manager = get_credential_manager()
        return self._credential_manager

    def register_tool(self, tool: ToolMeta) -> None:
        """Register a tool with the registry.

        If a tool with the same name already exists, it will be overwritten.

        Args:
            tool: ToolMeta instance containing tool metadata.
        """
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool from the registry.

        Args:
            name: Name of the tool to unregister.

        Returns:
            True if tool was removed, False if it wasn't registered.
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug("Unregistered tool: %s", name)
            return True
        return False

    def get_tool(self, name: str) -> ToolMeta | None:
        """Get tool metadata by name.

        Args:
            name: Name of the tool to look up.

        Returns:
            ToolMeta instance or None if not found.
        """
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: Name of the tool.

        Returns:
            True if the tool is registered.
        """
        return name in self._tools

    def list_tools(self, *, include_disabled: bool = False) -> list[ToolMeta]:
        """List all registered tools.

        Args:
            include_disabled: If True, include disabled tools.

        Returns:
            List of ToolMeta instances, sorted by name.
        """
        tools = list(self._tools.values())
        if not include_disabled:
            tools = [t for t in tools if t.enabled]
        return sorted(tools, key=lambda t: t.name)

    def check_credentials(self, tool_name: str) -> tuple[bool, list[str]]:
        """Check if all required credentials are available for a tool.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            Tuple of (all_available: bool, missing_credentials: list[str]).

        Raises:
            ToolNotFoundError: If the tool is not registered.
        """
        tool = self.get_tool(tool_name)
        if tool is None:
            raise ToolNotFoundError(tool_name)

        missing = []
        for service in tool.required_credentials:
            if not self.credential_manager.has_token(service):
                missing.append(service)

        return len(missing) == 0, missing

    def validate_credentials_for_tool(self, tool_name: str) -> None:
        """Validate that all required credentials are available.

        Args:
            tool_name: Name of the tool to validate.

        Raises:
            ToolNotFoundError: If the tool is not registered.
            MissingCredentialError: If any required credentials are missing.
        """
        all_available, missing = self.check_credentials(tool_name)
        if not all_available:
            raise MissingCredentialError(tool_name, missing)

    def check_health(self, name: str) -> tuple[bool, str]:
        """Run the health check for a specific tool.

        Args:
            name: Name of the tool to check.

        Returns:
            Tuple of (ok: bool, message: str).

        Raises:
            ToolNotFoundError: If the tool is not registered.
        """
        tool = self.get_tool(name)
        if tool is None:
            raise ToolNotFoundError(name)

        if not tool.enabled:
            return False, "Tool is disabled"

        try:
            return tool.health_check()
        except Exception as e:
            logger.error("Health check failed for %s: %s", name, e)
            return False, f"Health check error: {e}"

    def check_all_health(self) -> dict[str, tuple[bool, str]]:
        """Run health checks on all registered tools.

        Returns:
            Dict mapping tool names to (ok, message) tuples.
        """
        results = {}
        for name in self._tools:
            try:
                results[name] = self.check_health(name)
            except Exception as e:
                results[name] = (False, str(e))
        return results

    def get_tool_status(self, name: str) -> dict[str, Any]:
        """Get comprehensive status for a tool.

        Args:
            name: Name of the tool.

        Returns:
            Dict with tool status including health, credentials, and metadata.

        Raises:
            ToolNotFoundError: If the tool is not registered.
        """
        tool = self.get_tool(name)
        if tool is None:
            raise ToolNotFoundError(name)

        creds_available, missing_creds = self.check_credentials(name)
        health_ok, health_message = self.check_health(name)

        return {
            "name": tool.name,
            "description": tool.description,
            "version": tool.version,
            "enabled": tool.enabled,
            "capabilities": tool.capabilities,
            "required_credentials": tool.required_credentials,
            "credentials_available": creds_available,
            "missing_credentials": missing_creds,
            "health_ok": health_ok,
            "health_message": health_message,
            "ready": tool.enabled and creds_available and health_ok,
        }

    def get_all_status(self) -> list[dict[str, Any]]:
        """Get status for all registered tools.

        Returns:
            List of status dicts, sorted by tool name.
        """
        return [self.get_tool_status(name) for name in sorted(self._tools.keys())]


# Global tool registry instance
_tool_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance.

    Creates a new instance with built-in tools if one doesn't exist.

    Returns:
        The global ToolRegistry instance.
    """
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
        _register_builtin_tools(_tool_registry)
    return _tool_registry


def set_tool_registry(registry: ToolRegistry) -> None:
    """Set the global tool registry instance.

    Args:
        registry: The ToolRegistry instance to use globally.
    """
    global _tool_registry
    _tool_registry = registry


def reset_tool_registry() -> None:
    """Reset the global tool registry instance to the default state."""
    global _tool_registry
    _tool_registry = ToolRegistry()
    _register_builtin_tools(_tool_registry)


def _register_builtin_tools(registry: ToolRegistry) -> None:
    """Register built-in tools with the registry.

    Args:
        registry: The registry to register tools with.
    """
    # time_now - no credentials required, always healthy
    registry.register_tool(
        ToolMeta(
            name="time_now",
            description="Get the current time for a specified location",
            required_credentials=[],
            capabilities=["read", "time"],
            health_check=lambda: (True, "Time service available"),
            version="1.0.0",
        )
    )

    # weather_now - requires OpenWeatherMap API key
    def weather_health() -> tuple[bool, str]:
        from rex.credentials import get_credential_manager

        cm = get_credential_manager()
        if cm.has_token("openweathermap"):
            return True, "OpenWeatherMap API key configured"
        return False, "OpenWeatherMap API key not configured (set OPENWEATHERMAP_API_KEY)"

    registry.register_tool(
        ToolMeta(
            name="weather_now",
            description="Get current weather conditions for a location",
            required_credentials=["openweathermap"],
            capabilities=["read", "network", "weather"],
            health_check=weather_health,
            version="1.0.0",
            enabled=True,
        )
    )

    # web_search - stub, would require API key
    def web_search_health() -> tuple[bool, str]:
        from rex.credentials import get_credential_manager

        cm = get_credential_manager()
        # Check for any available search API
        for service in ["brave", "serpapi"]:
            if cm.has_token(service):
                return True, f"Using {service} for web search"
        return False, "No search API configured (need brave or serpapi)"

    registry.register_tool(
        ToolMeta(
            name="web_search",
            description="Search the web for information",
            required_credentials=[],  # Optional - uses any available search API
            capabilities=["read", "network", "search"],
            health_check=web_search_health,
            version="1.0.0",
            enabled=True,
        )
    )

    # send_email - stub, requires email token
    registry.register_tool(
        ToolMeta(
            name="send_email",
            description="Send an email message",
            required_credentials=["email"],
            capabilities=["write", "network", "email"],
            health_check=lambda: (False, "Email service not implemented"),
            version="1.0.0",
            enabled=True,
        )
    )

    # home_assistant - integration with Home Assistant
    def ha_health() -> tuple[bool, str]:
        from rex.credentials import get_credential_manager

        cm = get_credential_manager()
        if cm.has_token("home_assistant"):
            return True, "Home Assistant token configured"
        return False, "Home Assistant token not configured"

    registry.register_tool(
        ToolMeta(
            name="home_assistant",
            description="Control Home Assistant devices and scenes",
            required_credentials=["home_assistant"],
            capabilities=["read", "write", "network", "iot"],
            health_check=ha_health,
            version="1.0.0",
            enabled=True,
        )
    )

    logger.debug("Registered %d built-in tools", len(registry.list_tools(include_disabled=True)))


def register_tool(tool: ToolMeta) -> None:
    """Convenience function to register a tool with the global registry.

    Args:
        tool: ToolMeta instance to register.
    """
    get_tool_registry().register_tool(tool)


__all__ = [
    "ToolMeta",
    "ToolRegistry",
    "ToolNotFoundError",
    "MissingCredentialError",
    "HealthCheckFn",
    "get_tool_registry",
    "set_tool_registry",
    "reset_tool_registry",
    "register_tool",
]
