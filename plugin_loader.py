"""Dynamic discovery utilities for Rex plugins."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

from assistant_errors import PluginError
from logging_utils import get_logger

LOGGER = get_logger(__name__)


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError as exc:
            raise PluginError(f"Invalid manifest {manifest_path}: {exc}") from exc
    return {}


def load_plugins(path: str | os.PathLike[str] = "plugins") -> dict[str, Any]:
    """Dynamically import all plugin modules and call their ``register`` hooks."""

    base_path = Path(path)
    if not base_path.exists():
        LOGGER.warning("Plugin directory %s does not exist", base_path)
        return {}

    manifest = _load_manifest(base_path)
    capabilities: dict[str, Any] = {}

    for file in base_path.glob("*.py"):
        if file.name.startswith("_"):
            continue
        module_name = f"{base_path.name}.{file.stem}"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise PluginError(f"Failed to import plugin {module_name}: {exc}") from exc

        if not hasattr(module, "register"):
            LOGGER.info("Plugin %s has no register() function; skipping", module_name)
            continue

        try:
            result = module.register()
        except Exception as exc:
            raise PluginError(f"Plugin {module_name} raised during register: {exc}") from exc

        capabilities[module_name] = result

    if manifest.get("strict", False):
        expected = set(manifest.get("plugins", []))
        missing = expected.difference(capabilities.keys())
        if missing:
            raise PluginError(f"Missing required plugins: {sorted(missing)}")

    return capabilities


__all__ = ["load_plugins"]
