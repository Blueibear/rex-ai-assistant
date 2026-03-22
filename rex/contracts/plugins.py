"""Protocol definitions for Rex plugin loading and tool registration.

This contract captures the public APIs of ``rex.plugin_loader`` and
``rex.tool_registry`` so that OpenClaw-backed adapters can be substituted
transparently.

Two protocols are defined:

- ``PluginLoaderProtocol`` — discovers and registers plugin modules from a directory
- ``ToolRegistryProtocol`` — stores tool metadata and provides health/credential checks
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Plugin loading
# ---------------------------------------------------------------------------


@runtime_checkable
class PluginLoaderProtocol(Protocol):
    """Structural protocol for plugin discovery and loading.

    Covers the public API of ``rex.plugin_loader``:

    - ``load_plugins`` — scan a directory, import each module, and call its
      ``register()`` hook

    Implementations should raise a ``PluginError`` (or equivalent) for import
    failures and for missing required plugins when ``strict`` mode is active.
    """

    def load_plugins(
        self,
        path: str | os.PathLike[str] = "plugins",
    ) -> dict[str, Any]:
        """Discover and load plugins from *path*.

        Each ``.py`` file in the directory (excluding ``_``-prefixed files) is
        imported and its ``register()`` hook is called.  A ``manifest.json``
        file in the directory may declare required plugins (strict mode).

        Args:
            path: Filesystem path to the plugins directory.

        Returns:
            Mapping from module name to the value returned by its
            ``register()`` hook.

        Raises:
            ``PluginError`` (or equivalent): On import failure or unmet
            strict-mode requirements.
        """
        ...


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


@runtime_checkable
class ToolRegistryProtocol(Protocol):
    """Structural protocol for tool metadata storage and health reporting.

    Covers the full public API of ``rex.tool_registry.ToolRegistry``:

    - ``register_tool`` / ``unregister_tool`` — tool lifecycle management
    - ``get_tool`` / ``has_tool`` / ``list_tools`` — lookup and enumeration
    - ``check_credentials`` / ``validate_credentials_for_tool`` — credential gating
    - ``check_health`` / ``check_all_health`` — liveness probes
    - ``get_tool_status`` / ``get_all_status`` — rich status dicts

    Tool metadata objects (``ToolMeta``) are treated as ``Any`` here so this
    protocol remains dependency-free with respect to ``rex.tool_registry``.
    """

    def register_tool(self, tool: Any) -> None:
        """Register *tool* in the registry, overwriting any existing entry.

        Args:
            tool: A ``ToolMeta`` (or compatible) instance with at least a
                ``name`` (str) attribute.
        """
        ...

    def unregister_tool(self, name: str) -> bool:
        """Remove the tool named *name* from the registry.

        Args:
            name: The tool's unique name.

        Returns:
            ``True`` if the tool existed and was removed, ``False`` otherwise.
        """
        ...

    def get_tool(self, name: str) -> Any | None:
        """Return the ``ToolMeta`` for *name*, or ``None`` if not registered.

        Args:
            name: The tool's unique name.
        """
        ...

    def has_tool(self, name: str) -> bool:
        """Return ``True`` if a tool named *name* is registered."""
        ...

    def list_tools(self, *, include_disabled: bool = False) -> list[Any]:
        """Return all registered tools, sorted by name.

        Args:
            include_disabled: When ``True``, include tools whose ``enabled``
                flag is ``False``.

        Returns:
            List of ``ToolMeta`` (or compatible) instances.
        """
        ...

    def check_credentials(self, tool_name: str) -> tuple[bool, list[str]]:
        """Check whether all required credentials for *tool_name* are present.

        Args:
            tool_name: The tool's unique name.

        Returns:
            ``(all_available, missing_credentials)`` tuple.

        Raises:
            ``ToolNotFoundError`` (or equivalent): If the tool is not registered.
        """
        ...

    def validate_credentials_for_tool(self, tool_name: str) -> None:
        """Assert that all required credentials are present.

        Args:
            tool_name: The tool's unique name.

        Raises:
            ``ToolNotFoundError`` (or equivalent): If the tool is not registered.
            ``MissingCredentialError`` (or equivalent): If any credential is absent.
        """
        ...

    def check_health(self, name: str) -> tuple[bool, str]:
        """Run the registered health-check callable for *name*.

        Args:
            name: The tool's unique name.

        Returns:
            ``(ok, message)`` tuple.

        Raises:
            ``ToolNotFoundError`` (or equivalent): If the tool is not registered.
        """
        ...

    def check_all_health(self) -> dict[str, tuple[bool, str]]:
        """Run health checks for all registered tools.

        Returns:
            Mapping from tool name to ``(ok, message)`` tuple.
        """
        ...

    def get_tool_status(self, name: str) -> dict[str, Any]:
        """Return a rich status dict for *name*.

        The dict includes at minimum: ``name``, ``description``, ``version``,
        ``enabled``, ``capabilities``, ``required_credentials``,
        ``credentials_available``, ``missing_credentials``, ``health_ok``,
        ``health_message``, and ``ready``.

        Raises:
            ``ToolNotFoundError`` (or equivalent): If the tool is not registered.
        """
        ...

    def get_all_status(self) -> list[dict[str, Any]]:
        """Return status dicts for all registered tools, sorted by name."""
        ...
