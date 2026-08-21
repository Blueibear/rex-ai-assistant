"""Content-free observability for canonical output-routing decisions."""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any

from .models import ResolvedRoute

_LOGGER = logging.getLogger("rex.output_routing.service")
_AUDITED_MARKER = "_rex_output_routing_audited"


def record_route_decision(route: ResolvedRoute) -> ResolvedRoute:
    """Record non-sensitive decision metadata and return the route unchanged."""
    _LOGGER.info(
        "Output routing decision",
        extra={
            "event": "output_routing_decision",
            "output_kind": route.output_kind.value,
            "reason": route.reason,
            "has_target": route.target_id is not None,
            "fallback_mode": (
                route.fallback_mode.value if route.fallback_mode is not None else None
            ),
            "rule_index": route.rule_index,
            "suppressed": route.suppressed,
            "requires_confirmation": route.requires_confirmation,
            "volume_configured": route.target_volume is not None,
        },
    )
    return route


def install_output_routing_audit(service_class: type[Any]) -> None:
    """Wrap the canonical resolver once without changing its decision semantics."""
    original = service_class.resolve
    if getattr(original, _AUDITED_MARKER, False):
        return

    @wraps(original)
    def audited_resolve(self: Any, *args: Any, **kwargs: Any) -> ResolvedRoute:
        return record_route_decision(original(self, *args, **kwargs))

    setattr(audited_resolve, _AUDITED_MARKER, True)
    setattr(service_class, "resolve", audited_resolve)  # noqa: B010


__all__ = ["install_output_routing_audit", "record_route_decision"]
