"""Process-local access to output routing backed by the current media registry."""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock

from rex.media.registry import AudioTargetRegistry

from .service import OutputRoutingService

_lock = RLock()
_registry_provider: Callable[[], AudioTargetRegistry] | None = None


def set_output_registry_provider(
    provider: Callable[[], AudioTargetRegistry] | None,
) -> None:
    """Install the canonical current-registry provider used by routing callers."""
    global _registry_provider
    with _lock:
        _registry_provider = provider


def get_output_routing_service() -> OutputRoutingService:
    """Build a routing facade against the latest authorized target snapshot."""
    with _lock:
        provider = _registry_provider
    if provider is None:
        raise RuntimeError("Output routing is not configured")
    registry = provider()
    if not isinstance(registry, AudioTargetRegistry):
        raise RuntimeError("Output routing registry provider returned invalid state")
    return OutputRoutingService(registry)


__all__ = ["get_output_routing_service", "set_output_registry_provider"]