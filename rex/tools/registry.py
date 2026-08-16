"""Tool registry for Rex auto-dispatch (Phase 5 — US-TD-001).

Catalogs every tool Rex can invoke with metadata for automatic selection.
``available_tools()`` filters by which AppConfig fields are satisfied so
only truly callable tools are offered to the dispatcher.

Canonical implementation of ``ToolRegistryProtocol`` (see
``rex.tools.protocol``).  Current method names (``get``, ``all_tools``)
differ slightly from the protocol (``lookup``, ``list_tools``); they will be
aligned in a follow-up story without breaking existing callers.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from rex.capabilities.registry import Capability, CapabilityRegistry

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
        source: Where the tool executes — ``"local"`` for tools implemented
            in this repo, ``"openclaw"`` for tools dispatched through the
            OpenClaw gateway.  Added in US-011 to support cross-registry
            visibility.
    """

    name: str
    description: str
    capability_tags: list[str]
    requires_config: list[str]
    handler: Callable[..., Any]
    source: str = "local"
    operation: Literal["read", "mutation"] = "read"
    risk: Literal["safe", "sensitive", "prohibited"] = "safe"
    requires_identity: bool = False
    required_args: tuple[str, ...] = ()
    verifier: Callable[[dict[str, Any], Any], bool] | None = None
    input_schema: dict[str, str] = field(default_factory=dict)
    output_schema: dict[str, str] = field(default_factory=dict)
    required_permissions: tuple[str, ...] = ()
    health: str = "unknown"
    enabled: bool = True
    examples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Tool name cannot be empty")
        if not self.description:
            raise ValueError("Tool description cannot be empty")

    def to_capability(self) -> Capability:
        """Project executable tool metadata into the canonical Tool Card schema."""
        input_schema = dict(self.input_schema)
        if not input_schema:
            input_schema = dict.fromkeys(self.required_args, "any")
        return Capability(
            name=self.name,
            description=self.description,
            triggers=list(self.capability_tags),
            enabled=self.enabled and not self.requires_config,
            category="Tools",
            source=self.source,
            input_schema=input_schema,
            output_schema=dict(self.output_schema),
            required_permissions=self.required_permissions,
            health=self.health,
            operation=self.operation,
            risk=self.risk,
            verification_supported=self.verifier is not None,
            examples=self.examples,
            requires_identity=self.requires_identity,
            required_args=self.required_args,
            requires_config=tuple(self.requires_config),
        )


class ToolRegistry:
    """Registry of all Rex tools with config-based availability filtering.

    Usage::

        registry = ToolRegistry()
        registry.register(Tool(...))
        available = registry.available_tools(app_config)
    """

    def __init__(self, *, capability_registry: CapabilityRegistry | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._lock = threading.RLock()
        self._capability_registry = (
            capability_registry if capability_registry is not None else CapabilityRegistry()
        )

    @property
    def capability_registry(self) -> CapabilityRegistry:
        """Canonical metadata authority backing this executable registry."""
        return self._capability_registry

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, tool: Tool, *, replace: bool = False) -> Tool:
        """Bind a handler to canonical metadata without silent schema drift."""
        existing = self._tools.get(tool.name)
        if existing is not None and existing is not tool and not replace:
            if (
                existing.to_capability().static_signature()
                == tool.to_capability().static_signature()
            ):
                return existing
        self._capability_registry.register(tool.to_capability(), replace=replace)
        self._tools[tool.name] = tool
        logger.debug("tool_registry: registered %r", tool.name)
        return tool

    def register_remote_card(
        self, capability: Capability, *, handler: Callable[..., Any] | None = None
    ) -> Capability:
        """Adapt remote metadata without allowing it to rewrite local security policy."""
        resolved = self._capability_registry.register_remote(capability)
        if resolved is not capability:
            return resolved
        if capability.name not in self._tools and handler is not None:
            self._tools[capability.name] = Tool(
                name=capability.name,
                description=capability.description,
                capability_tags=list(capability.triggers),
                requires_config=list(capability.requires_config),
                handler=handler,
                source=capability.source,
                operation=capability.operation,
                risk=capability.risk,
                requires_identity=capability.requires_identity,
                required_args=capability.required_args,
                input_schema=dict(capability.input_schema),
                output_schema=dict(capability.output_schema),
                required_permissions=capability.required_permissions,
                health=capability.health,
                enabled=capability.enabled,
                examples=capability.examples,
            )
        return resolved

    def apply_openclaw_snapshot(
        self,
        capabilities: list[Capability] | tuple[Capability, ...],
        *,
        handler_factory: Callable[[str], Callable[..., Any]],
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Atomically project OpenClaw metadata into executable tool bindings.

        The canonical CapabilityRegistry remains the security authority. Remote
        cards are first applied there, then every remote card (including stale
        unavailable removals) is mirrored into this executable registry while
        holding one ToolRegistry mutation lock. Local tools are never replaced.
        """
        with self._lock:
            deltas = self._capability_registry.apply_openclaw_snapshot(capabilities)
            for card in self._capability_registry.list(include_disabled=True):
                if card.source != "openclaw":
                    continue
                existing = self._tools.get(card.id)
                if existing is not None and existing.source != "openclaw":
                    continue
                if existing is None and not card.enabled:
                    continue
                handler = handler_factory(card.id)
                self._tools[card.id] = Tool(
                    name=card.id,
                    description=card.description,
                    capability_tags=list(card.triggers),
                    requires_config=list(card.requires_config),
                    handler=handler,
                    source="openclaw",
                    operation=card.operation,
                    risk=card.risk,
                    requires_identity=card.requires_identity,
                    required_args=card.required_args,
                    input_schema=dict(card.input_schema),
                    output_schema=dict(card.output_schema),
                    required_permissions=card.required_permissions,
                    health=card.health,
                    enabled=card.enabled,
                    examples=card.examples,
                )
            return deltas

    def sync_openclaw_runtime_state(
        self, *, handler_factory: Callable[[str], Callable[..., Any]]
    ) -> None:
        """Mirror current canonical state into already-bound OpenClaw tools."""
        with self._lock:
            self._sync_openclaw_runtime_state_locked(handler_factory=handler_factory)

    def mark_openclaw_unavailable(
        self, *, handler_factory: Callable[[str], Callable[..., Any]]
    ) -> tuple[str, ...]:
        """Atomically disable canonical and executable OpenClaw state for readers."""
        with self._lock:
            changed = self._capability_registry.mark_openclaw_unavailable()
            self._sync_openclaw_runtime_state_locked(handler_factory=handler_factory)
            return changed

    def _sync_openclaw_runtime_state_locked(
        self, *, handler_factory: Callable[[str], Callable[..., Any]]
    ) -> None:
        for card in self._capability_registry.list(include_disabled=True):
            if card.source != "openclaw":
                continue
            existing = self._tools.get(card.id)
            if existing is None or existing.source != "openclaw":
                continue
            self._tools[card.id] = Tool(
                name=card.id,
                description=card.description,
                capability_tags=list(card.triggers),
                requires_config=list(card.requires_config),
                handler=handler_factory(card.id),
                source="openclaw",
                operation=card.operation,
                risk=card.risk,
                requires_identity=card.requires_identity,
                required_args=card.required_args,
                input_schema=dict(card.input_schema),
                output_schema=dict(card.output_schema),
                required_permissions=card.required_permissions,
                health=card.health,
                enabled=card.enabled,
                examples=card.examples,
            )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, name: str) -> Tool | None:
        """Return the tool with *name* or ``None`` if not registered."""
        with self._lock:
            return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        """Return all registered tools regardless of availability."""
        with self._lock:
            return list(self._tools.values())

    def list_tools(self) -> list[Any]:
        """Return ``ToolDescriptor`` objects for all registered tools.

        Implements the ``ToolRegistryProtocol.list_tools()`` interface.
        Returns descriptors (lazy import to avoid circular dependency with
        ``rex.tools.protocol``).
        """
        from rex.tools.protocol import ToolDescriptor  # local import to avoid cycles

        descriptors: list[Any] = []
        with self._lock:
            tools = sorted(self._tools.values(), key=lambda item: item.name)
        for tool in tools:
            card = self._capability_registry.get(tool.name)
            if card is None:
                continue
            descriptors.append(
                ToolDescriptor(
                    name=tool.name,
                    description=tool.description,
                    schema=dict(card.input_schema),
                    source=card.source,
                )
            )
        return descriptors

    def available_tools(
        self,
        config: Any,
        *,
        granted_permissions: set[str] | frozenset[str] | None = None,
    ) -> list[Tool]:
        """Return tools whose config and current permission requirements are satisfied.

        A field is *satisfied* when ``getattr(config, field_name, None)``
        is truthy.

        Args:
            config: An ``AppConfig`` instance (or any object with the
                expected attributes).

        Returns:
            Subset of registered tools that are fully configured.
        """
        result: list[Tool] = []
        with self._lock:
            tools = list(self._tools.values())
        for tool in tools:
            if not tool.enabled or not self._is_available(tool, config):
                continue
            if granted_permissions is not None and not self._capability_registry.is_authorized(
                tool.name, granted_permissions
            ):
                continue
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


def _web_search_handler(
    *,
    transcript: str = "",
    query: str = "",
    _runtime_config: Any = None,
    **kwargs: Any,
) -> Any:
    """Execute web search through the installed provider integration."""
    search_query = (query or transcript).strip()
    if not search_query:
        raise ValueError("web_search requires a non-empty query")

    try:
        from plugins.web_search import search_web
    except ImportError as exc:
        raise RuntimeError("web_search integration is not installed") from exc

    result = search_web(search_query, config=_runtime_config)
    if result is None:
        raise RuntimeError("web_search integration is not configured")
    return result


def _delegated_handler(tool_name: str) -> Callable[..., Any]:
    """Return a fail-closed handler for tools owned by a separate runtime path."""

    def _raise_delegation_error(**kwargs: Any) -> Any:
        raise RuntimeError(
            f"{tool_name} is not executable through the canonical tool registry; "
            "use its dedicated runtime handler"
        )

    return _raise_delegation_error


def _verify_numeric_field(field: str) -> Callable[[dict[str, Any], Any], bool]:
    def _verify(args: dict[str, Any], output: Any) -> bool:
        return isinstance(output, dict) and output.get(field) == int(args["level"])

    return _verify


def _verify_power_plan(args: dict[str, Any], output: Any) -> bool:
    return (
        isinstance(output, dict)
        and isinstance(output.get("power_plan"), str)
        and output["power_plan"].casefold() == str(args["name"]).casefold()
    )


def _build_default_registry(
    *, capability_registry: CapabilityRegistry | None = None
) -> ToolRegistry:
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
    from rex.tools.windows_repair import (
        check_disk_health,
        check_windows_update_status,
        flush_dns_cache,
        run_sfc_scan,
    )
    from rex.tools.windows_settings import (
        get_power_plan,
        get_volume,
        set_brightness,
        set_power_plan,
        set_volume,
    )

    registry = ToolRegistry(capability_registry=capability_registry)

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
            requires_config=["search_providers"],
            handler=_web_search_handler,
        )
    )

    registry.register(
        Tool(
            name="send_email",
            description="Send an email to one or more recipients.",
            capability_tags=["email", "messaging", "send"],
            requires_config=["email_accounts"],
            required_permissions=("email_send",),
            handler=send_email,
            operation="mutation",
            requires_identity=True,
            required_args=("to", "body"),
        )
    )

    registry.register(
        Tool(
            name="calendar_create",
            description="Create a new calendar event.",
            capability_tags=["calendar", "schedule", "event"],
            requires_config=[],
            handler=calendar_create,
            operation="mutation",
            requires_identity=True,
            required_args=("title", "start_time", "end_time"),
        )
    )

    registry.register(
        Tool(
            name="home_assistant_call_service",
            description="Control smart home devices via Home Assistant.",
            capability_tags=["smart_home", "home_assistant", "iot"],
            requires_config=["ha_base_url", "ha_token"],
            required_permissions=("ha_control",),
            handler=ha_call_service,
            operation="mutation",
            requires_identity=True,
            required_args=("domain", "service", "entity_id"),
        )
    )

    registry.register(
        Tool(
            name="send_sms",
            description="Send an SMS text message to a phone number.",
            capability_tags=["sms", "messaging", "text"],
            requires_config=[],
            required_permissions=("sms_send",),
            handler=send_sms,
            operation="mutation",
            requires_identity=True,
            required_args=("to", "body"),
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
            required_permissions=("computer_control",),
            handler=_read_file,
        )
    )

    registry.register(
        Tool(
            name="get_system_info",
            description="Get OS and hardware information (platform, CPU count, total RAM, boot time).",
            capability_tags=["windows", "diagnostics", "system"],
            requires_config=[],
            required_permissions=("computer_control",),
            handler=get_system_info,
        )
    )

    registry.register(
        Tool(
            name="get_cpu_usage",
            description="Get current CPU usage percentage and frequency.",
            capability_tags=["windows", "diagnostics", "cpu"],
            requires_config=[],
            required_permissions=("computer_control",),
            handler=get_cpu_usage,
        )
    )

    registry.register(
        Tool(
            name="get_memory_usage",
            description="Get current RAM and swap memory usage statistics.",
            capability_tags=["windows", "diagnostics", "memory", "ram"],
            requires_config=[],
            required_permissions=("computer_control",),
            handler=get_memory_usage,
        )
    )

    registry.register(
        Tool(
            name="get_disk_usage",
            description="Get disk usage for all mounted partitions.",
            capability_tags=["windows", "diagnostics", "disk", "storage"],
            requires_config=[],
            required_permissions=("computer_control",),
            handler=get_disk_usage,
        )
    )

    registry.register(
        Tool(
            name="get_battery_status",
            description="Get battery charge level and charging status.",
            capability_tags=["windows", "diagnostics", "battery"],
            requires_config=[],
            required_permissions=("computer_control",),
            handler=get_battery_status,
        )
    )

    registry.register(
        Tool(
            name="list_processes",
            description="List running processes sorted by CPU usage.",
            capability_tags=["windows", "diagnostics", "processes"],
            requires_config=[],
            required_permissions=("computer_control",),
            handler=list_processes,
        )
    )

    registry.register(
        Tool(
            name="get_volume",
            description="Get the current system master volume level (0–100).",
            capability_tags=["windows", "settings", "audio", "volume"],
            requires_config=[],
            required_permissions=("computer_control",),
            handler=get_volume,
        )
    )

    registry.register(
        Tool(
            name="set_volume",
            description="Set the system master volume level (0–100).",
            capability_tags=["windows", "settings", "audio", "volume"],
            requires_config=[],
            required_permissions=("computer_control",),
            handler=set_volume,
            operation="mutation",
            requires_identity=True,
            required_args=("level",),
            verifier=_verify_numeric_field("volume"),
        )
    )

    registry.register(
        Tool(
            name="set_brightness",
            description="Set the display brightness level (0–100).",
            capability_tags=["windows", "settings", "display", "brightness"],
            requires_config=[],
            required_permissions=("computer_control",),
            handler=set_brightness,
            operation="mutation",
            requires_identity=True,
            required_args=("level",),
            verifier=_verify_numeric_field("brightness"),
        )
    )

    registry.register(
        Tool(
            name="get_power_plan",
            description="Get the name of the currently active Windows power plan.",
            capability_tags=["windows", "settings", "power"],
            requires_config=[],
            required_permissions=("computer_control",),
            handler=get_power_plan,
        )
    )

    registry.register(
        Tool(
            name="set_power_plan",
            description="Switch the active Windows power plan by name (e.g. Balanced, High performance).",
            capability_tags=["windows", "settings", "power"],
            requires_config=[],
            required_permissions=("computer_control",),
            handler=set_power_plan,
            operation="mutation",
            requires_identity=True,
            required_args=("name",),
            verifier=_verify_power_plan,
        )
    )

    registry.register(
        Tool(
            name="check_disk_health",
            description="Check disk SMART health status and report any failure predictions.",
            capability_tags=["windows", "repair", "disk"],
            requires_config=[],
            required_permissions=("computer_control",),
            handler=check_disk_health,
        )
    )

    registry.register(
        Tool(
            name="check_windows_update_status",
            description="Check for pending Windows updates and list them.",
            capability_tags=["windows", "repair", "update"],
            requires_config=[],
            required_permissions=("computer_control",),
            handler=check_windows_update_status,
        )
    )

    registry.register(
        Tool(
            name="flush_dns_cache",
            description="Flush the Windows DNS resolver cache to fix name-resolution issues.",
            capability_tags=["windows", "repair", "dns", "network"],
            requires_config=[],
            required_permissions=("computer_control",),
            handler=flush_dns_cache,
            operation="mutation",
            requires_identity=True,
        )
    )

    registry.register(
        Tool(
            name="run_sfc_scan",
            description=(
                "Run System File Checker (sfc /scannow) to detect and repair corrupted "
                "Windows system files. Requires Administrator elevation and user confirmation."
            ),
            capability_tags=["windows", "repair", "sfc", "system"],
            requires_config=[],
            required_permissions=("computer_control",),
            handler=run_sfc_scan,
            operation="mutation",
            risk="sensitive",
            requires_identity=True,
        )
    )

    # Music Assistant tools (US-022)
    registry.register(
        Tool(
            name="music_play",
            description="Play music via Music Assistant. Args: query (str), room (str, optional).",
            capability_tags=["music", "play", "audio", "media"],
            requires_config=["music_assistant_url"],
            required_permissions=("ha_control",),
            handler=_delegated_handler("music_play"),
            operation="mutation",
            requires_identity=True,
        )
    )

    registry.register(
        Tool(
            name="music_pause",
            description="Pause music playback via Music Assistant. Args: room (str, optional).",
            capability_tags=["music", "pause", "audio", "media"],
            requires_config=["music_assistant_url"],
            required_permissions=("ha_control",),
            handler=_delegated_handler("music_pause"),
            operation="mutation",
            requires_identity=True,
        )
    )

    registry.register(
        Tool(
            name="music_resume",
            description="Resume paused music via Music Assistant. Args: room (str, optional).",
            capability_tags=["music", "resume", "audio", "media"],
            requires_config=["music_assistant_url"],
            required_permissions=("ha_control",),
            handler=_delegated_handler("music_resume"),
            operation="mutation",
            requires_identity=True,
        )
    )

    registry.register(
        Tool(
            name="music_skip",
            description="Skip to the next track via Music Assistant. Args: room (str, optional).",
            capability_tags=["music", "skip", "next", "audio", "media"],
            requires_config=["music_assistant_url"],
            required_permissions=("ha_control",),
            handler=_delegated_handler("music_skip"),
            operation="mutation",
            requires_identity=True,
        )
    )

    registry.register(
        Tool(
            name="music_volume",
            description=(
                "Set the volume level (0–100) via Music Assistant. "
                "Args: level (int), room (str, optional)."
            ),
            capability_tags=["music", "volume", "audio", "media"],
            requires_config=["music_assistant_url"],
            required_permissions=("ha_control",),
            handler=_delegated_handler("music_volume"),
            operation="mutation",
            requires_identity=True,
        )
    )

    return registry


#: Module-level singleton — lazily initialised on first call.
_default_registry: ToolRegistry | None = None


def ensure_default_registry(capability_registry: CapabilityRegistry) -> ToolRegistry:
    """Return the default executable registry bound to *capability_registry*."""
    global _default_registry
    if (
        _default_registry is None
        or _default_registry.capability_registry is not capability_registry
    ):
        _default_registry = _build_default_registry(capability_registry=capability_registry)
    return _default_registry


def get_default_registry() -> ToolRegistry:
    """Return the default executable registry backed by canonical metadata."""
    from rex.capabilities.registry import get_capability_registry  # noqa: PLC0415

    return ensure_default_registry(get_capability_registry())


def reset_default_registry() -> None:
    """Reset the executable singleton so tests cannot retain stale metadata."""
    global _default_registry
    _default_registry = None
