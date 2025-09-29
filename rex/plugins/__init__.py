"""Plugin loader for Rex."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import logging
import os
from dataclasses import dataclass
from typing import Iterable, List, Protocol

logger = logging.getLogger(__name__)


class Plugin(Protocol):
    name: str

    def initialize(self) -> None:
        ...

    def process(self, *args, **kwargs):
        ...

    def shutdown(self) -> None:
        ...


@dataclass
class PluginSpec:
    name: str
    plugin: Plugin


def load_plugins(path: str = "plugins") -> List[PluginSpec]:
    specs: List[PluginSpec] = []
    if not os.path.isdir(path):
        return specs

    for entry in os.listdir(path):
        if not entry.endswith(".py") or entry.startswith("_"):
            continue
        module_name = entry[:-3]
        if os.path.isabs(path):
            loader = importlib.machinery.SourceFileLoader(module_name, os.path.join(path, entry))
            spec = importlib.util.spec_from_loader(loader.name, loader)
            if spec is None or spec.loader is None:
                logger.warning("Could not load plugin from %s", entry)
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("Failed to exec plugin %s: %s", entry, exc)
                continue
            import_path = module_name
        else:
            module_path = f"{path.replace(os.sep, '.')}" if os.sep in path else path
            import_path = f"{module_path}.{module_name}" if module_path else module_name
            try:
                module = importlib.import_module(import_path)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("Failed to import plugin %s: %s", import_path, exc)
                continue

        register = getattr(module, "register", None)
        if register is None or not callable(register):
            logger.warning("Plugin %s does not expose register(); skipping", import_path)
            continue

        plugin_obj = register()
        if not hasattr(plugin_obj, "process"):
            logger.warning("Plugin %s returned invalid object", import_path)
            continue

        try:
            plugin_obj.initialize()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Plugin %s failed to initialise: %s", plugin_obj, exc)
            continue

        specs.append(PluginSpec(name=getattr(plugin_obj, "name", module_name), plugin=plugin_obj))
    return specs


def shutdown_plugins(specs: Iterable[PluginSpec]) -> None:
    for spec in specs:
        try:
            spec.plugin.shutdown()
        except Exception:  # pragma: no cover - defensive guard
            logger.exception("Error while shutting down plugin %s", spec.name)


__all__ = ["Plugin", "PluginSpec", "load_plugins", "shutdown_plugins"]
