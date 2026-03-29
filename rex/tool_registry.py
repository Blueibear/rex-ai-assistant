"""Thin wrapper around rex.openclaw.tool_registry that adds catalog validation.

The canonical ToolRegistry implementation lives in rex.openclaw.tool_registry
(relocated there during the OpenClaw HTTP migration).  This module re-exports
all public names for convenient top-level import AND validates tool registrations
against EXECUTABLE_TOOLS at registration time.

Usage::

    from rex.tool_registry import ToolRegistry, get_tool_registry
    # Identical to importing from rex.openclaw.tool_registry, but registration
    # logs a warning for tools outside the catalog.
"""

from __future__ import annotations

import logging

from rex.openclaw.tool_registry import (  # noqa: F401  re-exports
    HealthCheckFn,
    MissingCredentialError,
    ToolMeta,
    ToolNotFoundError,
    ToolRegistry,
    get_tool_registry,
    register_tool,
    reset_tool_registry,
    set_tool_registry,
)
from rex.tool_catalog import EXECUTABLE_TOOLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Catalog-validating subclass
# ---------------------------------------------------------------------------


class CatalogValidatingRegistry(ToolRegistry):
    """ToolRegistry that validates registrations against EXECUTABLE_TOOLS.

    Registering a tool whose name is NOT in EXECUTABLE_TOOLS logs a debug
    message (it is not an error — the registry may hold informational tools
    that are never directly executed by the router).

    Registering a tool whose name IS in EXECUTABLE_TOOLS logs at DEBUG level
    to confirm that the catalog is backed by a registered tool.
    """

    def register_tool(self, tool: ToolMeta) -> None:
        if tool.name in EXECUTABLE_TOOLS:
            logger.debug("Catalog tool registered: %s", tool.name)
        else:
            logger.debug(
                "Non-catalog tool registered: %s (not in EXECUTABLE_TOOLS, "
                "will not be routed by tool_router.execute_tool)",
                tool.name,
            )
        super().register_tool(tool)


__all__ = [
    "CatalogValidatingRegistry",
    "EXECUTABLE_TOOLS",
    "HealthCheckFn",
    "MissingCredentialError",
    "ToolMeta",
    "ToolNotFoundError",
    "ToolRegistry",
    "get_tool_registry",
    "register_tool",
    "reset_tool_registry",
    "set_tool_registry",
]
