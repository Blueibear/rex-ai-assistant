"""Capability registry and loader for profile based features."""

from __future__ import annotations

from typing import Callable

CapabilityLoader = Callable[[dict, dict], object | None]


def _load_local_commands(profile: dict, config: dict) -> object | None:
    return None


def _load_ha_router(profile: dict, config: dict) -> object | None:
    return None


def _load_web_search(profile: dict, config: dict) -> object | None:
    return None


def _load_plugins(profile: dict, config: dict) -> object | None:
    return None


CAPABILITY_REGISTRY: dict[str, CapabilityLoader] = {
    "local_commands": _load_local_commands,
    "ha_router": _load_ha_router,
    "web_search": _load_web_search,
    "plugins": _load_plugins,
}


def load_capabilities(profile: dict, config: dict) -> dict[str, object]:
    capabilities = profile.get("capabilities", []) if isinstance(profile, dict) else []
    enabled = []
    if isinstance(capabilities, list):
        enabled = [name for name in capabilities if isinstance(name, str) and name]

    loaded: dict[str, object] = {}
    for name in enabled:
        loader = CAPABILITY_REGISTRY.get(name)
        if loader is None:
            continue
        instance = loader(profile, config)
        loaded[name] = instance
    return loaded


__all__ = ["CAPABILITY_REGISTRY", "load_capabilities"]
