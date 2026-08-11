"""Per-interface provenance for one Assistant invocation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from rex.runtime.turn import TurnSource


@dataclass(frozen=True, slots=True)
class TurnInvocation:
    """Trusted edge metadata attached to the next Assistant turn."""

    source: TurnSource = TurnSource.ASSISTANT
    device_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", TurnSource(self.source))
        if self.device_id is not None and not self.device_id.strip():
            raise ValueError("device_id must be non-empty when supplied")


_DEFAULT_INVOCATION = TurnInvocation()
_CURRENT_INVOCATION: ContextVar[TurnInvocation] = ContextVar(
    "rex_turn_invocation", default=_DEFAULT_INVOCATION
)


def current_turn_invocation() -> TurnInvocation:
    """Return trusted provenance for the current execution context."""
    return _CURRENT_INVOCATION.get()


@contextmanager
def turn_invocation(
    source: TurnSource | str, *, device_id: str | None = None
) -> Iterator[TurnInvocation]:
    """Temporarily stamp Assistant calls with validated edge provenance."""
    invocation = TurnInvocation(source=TurnSource(source), device_id=device_id)
    token = _CURRENT_INVOCATION.set(invocation)
    try:
        yield invocation
    finally:
        _CURRENT_INVOCATION.reset(token)


__all__ = ["TurnInvocation", "current_turn_invocation", "turn_invocation"]
