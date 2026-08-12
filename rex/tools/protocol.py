"""Canonical typed interfaces for Rex's tool layer (US-010).

These Protocol classes define the contracts that the tool registry and
dispatcher must satisfy.  Concrete implementations live in
``rex.tools.registry`` and ``rex.tools.dispatcher``; OpenClaw adapter
classes in ``rex.openclaw`` implement the same interfaces as thin adapters.

Downstream code that needs to call tools should depend on these protocols
rather than the concrete classes so that adapters can be swapped without
changing call sites.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from rex.actions.lifecycle import ActionLifecycleSnapshot

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ToolDescriptor:
    """Metadata about a single registered tool.

    Attributes:
        name:        Unique tool identifier, e.g. ``"web_search"``.
        description: Human-readable description of what the tool does.
        schema:      JSON-Schema dict describing the tool's arguments.
        source:      Where the tool is executed — ``"local"`` for tools
                     implemented in this repo, ``"openclaw"`` for tools
                     dispatched through the OpenClaw gateway.
    """

    name: str
    description: str
    schema: dict = field(default_factory=dict)
    source: str = "local"


@dataclass
class ToolResult:
    """Result of a single tool execution.

    Attributes:
        success: ``True`` for a completed read or independently verified mutation.
        output:  Return value of the tool handler (any JSON-serialisable
                 type).  ``None`` when *success* is ``False``.
        error:   Human-readable execution error, if any.
        status:  Normalized lifecycle outcome. Mutations are never marked
                 successful merely because their handler returned.
    """

    success: bool
    output: Any = None
    error: str | None = None
    status: str = "completed"
    detail: str | None = None
    request_id: str | None = None
    risk: str = "safe"
    stages: tuple[str, ...] = ()
    lifecycle: ActionLifecycleSnapshot | None = None


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ToolRegistryProtocol(Protocol):
    """Interface that every tool registry must implement.

    The canonical implementation is ``rex.tools.registry.ToolRegistry``.
    OpenClaw's adapter (``rex.openclaw.tool_registry``) delegates to it and
    adds gateway-specific metadata alongside.
    """

    def register(self, name: str, fn: Callable[..., Any], schema: dict) -> None:
        """Register (or replace) a tool with the given *name*.

        Args:
            name:   Unique tool identifier.
            fn:     Callable to invoke when the tool is dispatched.
            schema: JSON-Schema dict describing ``fn``'s arguments.
        """
        ...

    def lookup(self, name: str) -> Callable[..., Any] | None:
        """Return the handler for *name*, or ``None`` if not registered."""
        ...

    def list_tools(self) -> list[ToolDescriptor]:
        """Return descriptors for all registered tools."""
        ...


@runtime_checkable
class ToolDispatcherProtocol(Protocol):
    """Interface that every tool dispatcher must implement.

    The canonical implementation is ``rex.tools.dispatcher.ToolDispatcher``.
    """

    def dispatch(
        self,
        name: str,
        args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Execute the tool identified by *name* with *args*.

        Args:
            name:    Tool identifier (must be registered in the registry).
            args:    Keyword arguments forwarded to the tool handler.
            context: Optional ambient context (caller identity, session id,
                     etc.) that the handler may inspect but should not modify.

        Returns:
            A :class:`ToolResult` describing success or failure.
        """
        ...
