"""Process-local access to output routing backed by the current media registry."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from threading import RLock
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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


def user_local_now(user_id: str) -> datetime:
    """Return current time in the user's configured timezone, or UTC safely."""
    timezone_name: str | None = None
    try:
        from rex.user_profile_service import UserProfileService

        profile = UserProfileService().get_profile(user_id)
        candidate = profile.preferences.get("timezone")
        if isinstance(candidate, str) and candidate.strip():
            timezone_name = candidate.strip()
    except Exception:
        timezone_name = None
    if not timezone_name:
        try:
            from rex.geolocation import get_cached_timezone

            timezone_name = get_cached_timezone()
        except Exception:
            timezone_name = None
    try:
        zone = ZoneInfo(timezone_name or "UTC")
    except ZoneInfoNotFoundError:
        zone = ZoneInfo("UTC")
    return datetime.now(UTC).astimezone(zone)


__all__ = [
    "get_output_routing_service",
    "set_output_registry_provider",
    "user_local_now",
]
