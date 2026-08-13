"""Privacy-safe progressive status derived only from canonical turn events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from rex.runtime.events import EventKind, TurnEvent


class TurnStatus(StrEnum):
    """Small user-facing status vocabulary shared by every interface."""

    THINKING = "thinking"
    CHECKING = "checking"
    ACTING = "acting"
    VERIFYING = "verifying"
    SPEAKING = "speaking"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class TurnStatusUpdate:
    """A deliberately content-free projection safe for UI/wire surfaces."""

    turn_id: str
    sequence: int
    status: TurnStatus
    terminal: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_id": self.turn_id,
            "sequence": self.sequence,
            "status": self.status.value,
            "terminal": self.terminal,
        }


def _action_is_verification(event: TurnEvent) -> bool:
    stage = str(event.details.get("stage") or "").casefold()
    state = str(event.details.get("state") or event.details.get("status") or "").casefold()
    return "verif" in stage or state in {"verified", "unverified", "attempted_unverified"}


def project_turn_status(event: TurnEvent) -> TurnStatusUpdate:
    """Project one canonical event without forwarding any event details or identity."""
    if event.kind is EventKind.TURN_STARTED:
        status = TurnStatus.THINKING
    elif event.kind in {
        EventKind.CONTEXT_PROGRESS,
        EventKind.ROUTE_PROGRESS,
        EventKind.CAPABILITY_PROGRESS,
    }:
        status = TurnStatus.CHECKING
    elif event.kind is EventKind.MODEL_PROGRESS:
        status = TurnStatus.THINKING
    elif event.kind is EventKind.ACTION_PROGRESS:
        status = TurnStatus.VERIFYING if _action_is_verification(event) else TurnStatus.ACTING
    elif event.kind is EventKind.RESPONSE_PROGRESS:
        stage = str(event.details.get("stage") or "").casefold()
        status = TurnStatus.VERIFYING if stage == "output_validation" else TurnStatus.SPEAKING
    elif event.kind is EventKind.COMPLETED:
        status = TurnStatus.DONE
    elif event.kind is EventKind.CANCELLED:
        status = TurnStatus.CANCELLED
    else:
        status = TurnStatus.ERROR
    return TurnStatusUpdate(
        turn_id=event.turn_id,
        sequence=event.sequence,
        status=status,
        terminal=event.is_terminal,
    )


StatusSink = Callable[[TurnStatusUpdate], None]


class TurnStatusProjector:
    """Observer adapter that suppresses presentation-only duplicate states."""

    def __init__(self, sink: StatusSink) -> None:
        self._sink = sink
        self._last_by_turn: dict[str, TurnStatus] = {}

    def observe(self, event: TurnEvent) -> None:
        update = project_turn_status(event)
        previous = self._last_by_turn.get(update.turn_id)
        if update.terminal or update.status is not previous:
            self._sink(update)
        if update.terminal:
            self._last_by_turn.pop(update.turn_id, None)
        else:
            self._last_by_turn[update.turn_id] = update.status


__all__ = [
    "StatusSink",
    "TurnStatus",
    "TurnStatusProjector",
    "TurnStatusUpdate",
    "project_turn_status",
]
