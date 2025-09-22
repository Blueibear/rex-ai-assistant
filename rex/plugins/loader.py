"""Dynamic plugin loading utilities."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict

from ..config import settings
from ..logging_utils import configure_logger
from .base import Plugin

LOGGER = configure_logger(__name__)


def load_plugins(package: str = "plugins") -> Dict[str, Plugin]:
    """Import all plugin modules and return instantiated plugins."""

    discovered: Dict[str, Plugin] = {}
    try:
        pkg = importlib.import_module(package)
    except ModuleNotFoundError:
        LOGGER.debug("Plugin package %s not found", package)
        return discovered

    if not hasattr(pkg, "__path__"):
        LOGGER.debug("Package %s has no __path__ attribute", package)
        return discovered

    for module_info in pkgutil.iter_modules(pkg.__path__):
        module_name = f"{package}.{module_info.name}"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - plugin import errors
            LOGGER.error("Failed to import plugin %s: %s", module_name, exc)
            continue

        plugin = getattr(module, "PLUGIN", None)
        if plugin is None:
            LOGGER.debug("Module %s does not expose a PLUGIN instance", module_name)
            continue
        if not all(hasattr(plugin, attr) for attr in ("initialise", "process", "shutdown", "name")):
            LOGGER.debug("Plugin %s is missing required hooks", module_name)
            continue

        try:
            plugin.initialise()
        except Exception as exc:  # pragma: no cover - plugin init errors
            LOGGER.error("Plugin %s failed to initialise: %s", module_name, exc)
            continue

        discovered[plugin.name] = plugin
        LOGGER.info("Loaded plugin %s", plugin.name)

    if not settings.enable_search_plugin and "web_search" in discovered:
        discovered.pop("web_search")
        LOGGER.info("Web search plugin disabled via configuration")

    return discovered


def shutdown_plugins(plugins: Dict[str, Plugin]) -> None:
    for plugin in plugins.values():
        try:
            plugin.shutdown()
        except Exception as exc:  # pragma: no cover - plugin shutdown errors
            LOGGER.warning("Plugin %s failed to shut down cleanly: %s", plugin.name, exc)


__all__ = ["load_plugins", "shutdown_plugins"]
