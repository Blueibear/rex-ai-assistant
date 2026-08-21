"""Immutable context contracts for one assistant turn."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from rex.identity import validate_user_id
from rex.runtime.cancellation import TurnCancellation


class TurnScope(StrEnum):
    """Data/authority scope associated with a turn."""

    USER = "user"
    HOUSEHOLD = "household"


class TurnSource(StrEnum):
    """Interface that originated a turn."""

    ASSISTANT = "assistant"
    CLI = "cli"
    ELECTRON = "electron"
    VOICE = "voice"
    MOBILE = "mobile"
    API = "api"
    TELEGRAM = "telegram"
    TELEPHONY = "telephony"
    MQTT = "mqtt"


class IdentityResolution(StrEnum):
    """Trusted provenance for how the turn's user identity was resolved."""

    EXPLICIT = "explicit"
    VOICE_RECOGNIZED = "voice_recognized"
    VOICE_REVIEW = "voice_review"
    FALLBACK = "fallback"
    UNKNOWN = "unknown"


class ResponseMode(StrEnum):
    """Delivery mode requested by the originating surface."""

    VOICE = "voice"
    SCREEN = "screen"
    HYBRID = "hybrid"
    AUTOMATION = "automation"


@dataclass(frozen=True, slots=True)
class AuthorizationSnapshotRef:
    """Immutable references to policy and permission snapshots."""

    policy_ref: str
    permission_ref: str

    def __post_init__(self) -> None:
        if not self.policy_ref.strip():
            raise ValueError("policy_ref must not be empty")
        if not self.permission_ref.strip():
            raise ValueError("permission_ref must not be empty")


@dataclass(frozen=True, slots=True)
class TurnContext:
    """Validated identity, timing, and delivery context for one turn."""

    turn_id: str
    user_id: str
    scope: TurnScope
    source: TurnSource
    device_id: str | None
    response_mode: ResponseMode
    authorization: AuthorizationSnapshotRef
    cancellation: TurnCancellation
    started_monotonic_ns: int
    deadline_monotonic_ns: int | None = None
    identity_resolution: IdentityResolution = IdentityResolution.EXPLICIT

    def __post_init__(self) -> None:
        validate_user_id(self.user_id)
        try:
            object.__setattr__(self, "scope", TurnScope(self.scope))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid turn scope: {self.scope!r}") from exc
        try:
            object.__setattr__(self, "source", TurnSource(self.source))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid turn source: {self.source!r}") from exc
        try:
            object.__setattr__(
                self,
                "identity_resolution",
                IdentityResolution(self.identity_resolution),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid identity resolution: {self.identity_resolution!r}") from exc
        try:
            object.__setattr__(self, "response_mode", ResponseMode(self.response_mode))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid response mode: {self.response_mode!r}") from exc
        if not self.turn_id.strip():
            raise ValueError("turn_id must not be empty")
        if self.cancellation.turn_id != self.turn_id:
            raise ValueError("cancellation scope does not belong to turn")
        if self.started_monotonic_ns < 0:
            raise ValueError("started_monotonic_ns must be non-negative")
        if (
            self.deadline_monotonic_ns is not None
            and self.deadline_monotonic_ns < self.started_monotonic_ns
        ):
            raise ValueError("deadline_monotonic_ns precedes turn start")
        if self.device_id is not None and not self.device_id.strip():
            raise ValueError("device_id must be non-empty when supplied")

    @classmethod
    def create(
        cls,
        *,
        user_id: str,
        scope: TurnScope | str,
        source: TurnSource | str,
        device_id: str | None,
        response_mode: ResponseMode | str,
        authorization: AuthorizationSnapshotRef,
        identity_resolution: IdentityResolution | str | None = None,
        timeout_seconds: float | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> TurnContext:
        """Create a validated context using a monotonic deadline.

        When trusted edge provenance is active and no explicit resolution is
        supplied, inherit it from ``rex.runtime.invocation``. Direct library
        construction remains explicitly identified by default.
        """
        validated_user = validate_user_id(user_id)
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        if identity_resolution is None:
            # Lazy import avoids a module-load cycle: invocation imports the
            # enums defined in this module.
            from rex.runtime.invocation import current_turn_invocation

            identity_resolution = current_turn_invocation().identity_resolution
        turn_id = uuid.uuid4().hex
        started_ns = clock()
        deadline_ns = None
        if timeout_seconds is not None:
            deadline_ns = started_ns + int(timeout_seconds * 1_000_000_000)
        return cls(
            turn_id=turn_id,
            user_id=validated_user,
            scope=TurnScope(scope),
            source=TurnSource(source),
            device_id=device_id,
            response_mode=ResponseMode(response_mode),
            authorization=authorization,
            cancellation=TurnCancellation(turn_id),
            started_monotonic_ns=started_ns,
            deadline_monotonic_ns=deadline_ns,
            identity_resolution=IdentityResolution(identity_resolution),
        )
