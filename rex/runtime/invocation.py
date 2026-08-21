"""Per-interface provenance for one Assistant invocation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from rex.runtime.turn import IdentityResolution, TurnSource


@dataclass(frozen=True, slots=True)
class TurnInvocation:
    """Trusted edge metadata attached to the next Assistant turn."""

    source: TurnSource = TurnSource.ASSISTANT
    device_id: str | None = None
    identity_resolution: IdentityResolution = IdentityResolution.EXPLICIT

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", TurnSource(self.source))
        object.__setattr__(
            self,
            "identity_resolution",
            IdentityResolution(self.identity_resolution),
        )
        if self.device_id is not None and not self.device_id.strip():
            raise ValueError("device_id must be non-empty when supplied")


_DEFAULT_INVOCATION = TurnInvocation()
_CURRENT_INVOCATION: ContextVar[TurnInvocation] = ContextVar(
    "rex_turn_invocation", default=_DEFAULT_INVOCATION
)
_STAGED_IDENTITY_RESOLUTION: ContextVar[IdentityResolution] = ContextVar(
    "rex_staged_identity_resolution",
    default=IdentityResolution.UNKNOWN,
)


def current_turn_invocation() -> TurnInvocation:
    """Return trusted provenance for the current execution context."""
    return _CURRENT_INVOCATION.get()


def stage_identity_resolution(
    resolution: IdentityResolution | str,
) -> IdentityResolution:
    """Stage trusted voice-identity provenance for the next voice invocation.

    This function is called only by trusted runtime/voice adapters. It carries
    provenance, not authority; the resulting user ID is still validated and
    all downstream permission checks remain in force.
    """
    normalized = IdentityResolution(resolution)
    _STAGED_IDENTITY_RESOLUTION.set(normalized)
    return normalized


def clear_staged_identity_resolution() -> None:
    """Clear any staged voice provenance so stale identity cannot be reused."""
    _STAGED_IDENTITY_RESOLUTION.set(IdentityResolution.UNKNOWN)


@contextmanager
def turn_invocation(
    source: TurnSource | str,
    *,
    device_id: str | None = None,
    identity_resolution: IdentityResolution | str | None = None,
) -> Iterator[TurnInvocation]:
    """Temporarily stamp Assistant calls with validated edge provenance."""
    normalized_source = TurnSource(source)
    if identity_resolution is None:
        if normalized_source is TurnSource.VOICE:
            normalized_identity = _STAGED_IDENTITY_RESOLUTION.get()
        else:
            normalized_identity = IdentityResolution.EXPLICIT
    else:
        normalized_identity = IdentityResolution(identity_resolution)

    invocation = TurnInvocation(
        source=normalized_source,
        device_id=device_id,
        identity_resolution=normalized_identity,
    )
    token = _CURRENT_INVOCATION.set(invocation)
    try:
        yield invocation
    finally:
        _CURRENT_INVOCATION.reset(token)
        if normalized_source is TurnSource.VOICE:
            # Voice provenance is one-shot. A later turn must be restamped by
            # the trusted identity adapter rather than inheriting stale state.
            clear_staged_identity_resolution()


__all__ = [
    "TurnInvocation",
    "clear_staged_identity_resolution",
    "current_turn_invocation",
    "stage_identity_resolution",
    "turn_invocation",
]
