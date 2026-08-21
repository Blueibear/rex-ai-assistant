"""Output routing for due timer and alarm events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from rex.timekeeping.models import DueEvent, utc_now

from .models import OutputKind
from .service import OutputRoutingService


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    """Truthful result of selecting and attempting a due-event output."""

    delivered: bool
    target_id: str | None
    reason: str
    target_volume: int | None = None


class OutputDeliveryService:
    """Resolve current per-user routing at fire time and invoke one sender."""

    def __init__(
        self,
        routing: OutputRoutingService,
        *,
        sender: Callable[[str, DueEvent], bool],
        now_func: Callable[[], datetime] | None = None,
    ) -> None:
        self._routing = routing
        self._sender = sender
        self._now = now_func or utc_now

    def deliver_due_event(self, event: DueEvent) -> DeliveryResult:
        if not isinstance(event, DueEvent):
            raise TypeError("event must be a DueEvent")
        kind = OutputKind.TIMER if event.kind == "timer" else OutputKind.ALARM
        route = self._routing.resolve(
            user_id=event.user_id,
            output_kind=kind,
            explicit_target_text=event.output_target_id,
            origin_device_id=None,
            at=self._now(),
        )
        if route.target_id is None or route.suppressed:
            return DeliveryResult(
                delivered=False,
                target_id=route.target_id,
                reason=route.reason,
                target_volume=route.target_volume,
            )
        try:
            delivered = bool(self._sender(route.target_id, event))
        except Exception:
            delivered = False
        return DeliveryResult(
            delivered=delivered,
            target_id=route.target_id,
            reason=route.reason if delivered else "delivery_failed",
            target_volume=route.target_volume,
        )


__all__ = ["DeliveryResult", "OutputDeliveryService"]
