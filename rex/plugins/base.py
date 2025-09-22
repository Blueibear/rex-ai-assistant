"""Plugin interfaces for Rex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class PluginContext:
    """Context passed to plugins when they are executed."""

    user_id: str
    text: str


class Plugin(Protocol):
    """Plugins provide optional capabilities such as web search."""

    name: str

    def initialise(self) -> None:
        """Called once when the plugin is loaded."""

    def process(self, context: PluginContext) -> str | None:
        """Return a string result or ``None`` if nothing is produced."""

    def shutdown(self) -> None:
        """Called during application shutdown to free resources."""


__all__ = ["Plugin", "PluginContext"]
