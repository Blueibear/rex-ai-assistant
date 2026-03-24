"""In-memory tool-call cache for the Rex autonomy engine.

:class:`ToolCache` prevents redundant tool invocations within a single plan
run by memoising call results keyed on ``(tool_name, sorted_args)``.

The cache is intentionally *not* shared across plan runs — callers must
create a new :class:`ToolCache` instance for each :class:`~rex.autonomy.models.Plan`
execution.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ToolCache
# ---------------------------------------------------------------------------


class ToolCache:
    """In-memory cache for tool call results, scoped to one plan run.

    Cache key: ``(tool_name, frozenset(sorted(args.items())))``

    Usage::

        cache = ToolCache()

        result = cache.get("search", {"query": "hello"})
        if result is None:
            result = search_tool(query="hello")
            cache.set("search", {"query": "hello"}, result)

    The cache does **not** persist between :class:`~rex.autonomy.models.Plan`
    executions — a new instance should be created for each plan run.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, frozenset[tuple[str, Any]]], str] = {}

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(tool: str, args: dict[str, Any]) -> tuple[str, frozenset[tuple[str, Any]]]:
        return (tool, frozenset(sorted(args.items())))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, tool: str, args: dict[str, Any]) -> str | None:
        """Return the cached result for *(tool, args)*, or ``None`` on a miss.

        A cache hit is logged at ``DEBUG`` level.

        Args:
            tool: Tool name.
            args: Keyword arguments passed to the tool.

        Returns:
            The previously cached result string, or ``None`` if not cached.
        """
        key = self._key(tool, args)
        result = self._store.get(key)
        if result is not None:
            logger.debug("Tool cache hit: %s(%s)", tool, args)
        return result

    def set(self, tool: str, args: dict[str, Any], result: str) -> None:
        """Store *result* for the given *(tool, args)* key.

        Args:
            tool: Tool name.
            args: Keyword arguments passed to the tool.
            result: The string result returned by the tool.
        """
        key = self._key(tool, args)
        self._store[key] = result

    def __len__(self) -> int:
        """Return the number of cached entries."""
        return len(self._store)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["ToolCache"]
