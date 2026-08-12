"""Canonical verification-first lifecycle for Rex actions."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import StrEnum


class ActionState(StrEnum):
    """Canonical action states shared by every execution adapter."""

    PLANNED = "planned"
    AUTHORIZED = "authorized"
    ATTEMPTED = "attempted"
    COMPLETED = "completed"
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Completed is a truthful terminal state for read-only work and may also be an
# intermediate state for a mutation with independent verification evidence.
# Unverified is terminal uncertainty: it must never transition to verified
# without creating a new action attempt/correlation chain.
ALLOWED_TRANSITIONS: dict[ActionState, frozenset[ActionState]] = {
    ActionState.PLANNED: frozenset(
        {ActionState.AUTHORIZED, ActionState.FAILED, ActionState.CANCELLED}
    ),
    ActionState.AUTHORIZED: frozenset(
        {ActionState.ATTEMPTED, ActionState.FAILED, ActionState.CANCELLED}
    ),
    ActionState.ATTEMPTED: frozenset(
        {
            ActionState.COMPLETED,
            ActionState.UNVERIFIED,
            ActionState.FAILED,
            ActionState.CANCELLED,
        }
    ),
    ActionState.COMPLETED: frozenset(
        {ActionState.VERIFIED, ActionState.UNVERIFIED, ActionState.FAILED}
    ),
    ActionState.VERIFIED: frozenset(),
    ActionState.UNVERIFIED: frozenset(),
    ActionState.FAILED: frozenset(),
    ActionState.CANCELLED: frozenset(),
}


class InvalidActionTransition(RuntimeError):
    """Raised when an action attempts an invalid or terminal transition."""


@dataclass(frozen=True)
class ActionCorrelation:
    """Immutable deterministic identifiers linking all action evidence."""

    action_id: str
    plan_id: str | None
    attempt_id: str
    verification_id: str
    audit_id: str
    user_result_id: str

    @classmethod
    def create(cls, action_id: str, *, plan_id: str | None = None) -> ActionCorrelation:
        canonical = action_id.strip()
        if not canonical:
            raise ValueError("action_id is required")
        normalized_plan = plan_id.strip() if isinstance(plan_id, str) and plan_id.strip() else None
        return cls(
            action_id=canonical,
            plan_id=normalized_plan,
            attempt_id=f"attempt:{canonical}",
            verification_id=f"verify:{canonical}",
            audit_id=f"audit:{canonical}",
            user_result_id=f"result:{canonical}",
        )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "action_id": self.action_id,
            "plan_id": self.plan_id,
            "attempt_id": self.attempt_id,
            "verification_id": self.verification_id,
            "audit_id": self.audit_id,
            "user_result_id": self.user_result_id,
        }


@dataclass(frozen=True)
class ActionTransition:
    state: ActionState
    sequence: int
    evidence_ref: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "sequence": self.sequence,
            "evidence_ref": self.evidence_ref,
        }


@dataclass(frozen=True)
class ActionLifecycleSnapshot:
    correlation: ActionCorrelation
    state: ActionState
    transitions: tuple[ActionTransition, ...]

    @property
    def success(self) -> bool:
        return self.state in {ActionState.COMPLETED, ActionState.VERIFIED}

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "success": self.success,
            "correlation": self.correlation.to_dict(),
            "transitions": [item.to_dict() for item in self.transitions],
        }


class ActionLifecycle:
    """Small thread-safe state machine that fails closed on invalid transitions."""

    def __init__(self, correlation: ActionCorrelation) -> None:
        self._correlation = correlation
        self._state = ActionState.PLANNED
        self._transitions: list[ActionTransition] = [
            ActionTransition(ActionState.PLANNED, sequence=0, evidence_ref="action:planned")
        ]
        self._lock = threading.Lock()

    @classmethod
    def create(cls, *, action_id: str, plan_id: str | None = None) -> ActionLifecycle:
        return cls(ActionCorrelation.create(action_id, plan_id=plan_id))

    @property
    def correlation(self) -> ActionCorrelation:
        return self._correlation

    @property
    def state(self) -> ActionState:
        with self._lock:
            return self._state

    def transition(
        self, state: ActionState | str, *, evidence_ref: str | None = None
    ) -> ActionLifecycleSnapshot:
        target = ActionState(state)
        with self._lock:
            allowed = ALLOWED_TRANSITIONS[self._state]
            if target not in allowed:
                raise InvalidActionTransition(
                    f"invalid action transition: {self._state.value} -> {target.value}"
                )
            self._state = target
            self._transitions.append(
                ActionTransition(target, sequence=len(self._transitions), evidence_ref=evidence_ref)
            )
            return self._snapshot_unlocked()

    def snapshot(self) -> ActionLifecycleSnapshot:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> ActionLifecycleSnapshot:
        return ActionLifecycleSnapshot(
            correlation=self._correlation,
            state=self._state,
            transitions=tuple(self._transitions),
        )


def lifecycle_from_legacy_status(
    status: str, *, action_id: str, plan_id: str | None = None
) -> ActionLifecycleSnapshot:
    """Adapt existing public result vocabularies into the canonical lifecycle."""
    lifecycle = ActionLifecycle.create(action_id=action_id, plan_id=plan_id)
    normalized = str(status or "").strip().lower()

    if normalized == "verified":
        lifecycle.transition(ActionState.AUTHORIZED, evidence_ref="policy:authorized")
        lifecycle.transition(ActionState.ATTEMPTED, evidence_ref="execution:attempted")
        lifecycle.transition(ActionState.COMPLETED, evidence_ref="execution:completed")
        return lifecycle.transition(ActionState.VERIFIED, evidence_ref="verification:verified")
    if normalized == "completed":
        lifecycle.transition(ActionState.AUTHORIZED, evidence_ref="policy:authorized")
        lifecycle.transition(ActionState.ATTEMPTED, evidence_ref="execution:attempted")
        return lifecycle.transition(ActionState.COMPLETED, evidence_ref="execution:completed")
    if normalized in {"attempted_unverified", "unverified"}:
        lifecycle.transition(ActionState.AUTHORIZED, evidence_ref="policy:authorized")
        lifecycle.transition(ActionState.ATTEMPTED, evidence_ref="execution:attempted")
        return lifecycle.transition(ActionState.UNVERIFIED, evidence_ref="verification:unverified")
    if normalized == "cancelled":
        return lifecycle.transition(ActionState.CANCELLED, evidence_ref="action:cancelled")
    if normalized == "confirmation_required":
        return lifecycle.snapshot()
    # Denied, unavailable, explicit failure, and unknown statuses all fail closed.
    return lifecycle.transition(ActionState.FAILED, evidence_ref="action:failed")


def render_action_outcome(
    lifecycle: ActionLifecycleSnapshot,
    subject: str,
    *,
    detail: str | None = None,
) -> str:
    """Render action wording from canonical evidence rather than exceptions."""
    label = subject.strip() or "the action"
    state = lifecycle.state
    if state is ActionState.VERIFIED:
        return detail or f"Verified {label}."
    if state is ActionState.COMPLETED:
        return detail or f"Completed {label}."
    if state is ActionState.UNVERIFIED:
        base = f"I attempted {label}, but could not verify the result."
        return f"{base} {detail}" if detail else base
    if state is ActionState.CANCELLED:
        return detail or f"{label} was cancelled."
    if state is ActionState.FAILED:
        return detail or f"{label} failed."
    if state is ActionState.AUTHORIZED:
        return detail or f"{label} is authorized but has not been attempted."
    if state is ActionState.ATTEMPTED:
        return detail or f"{label} was attempted, but no final outcome is available."
    return detail or f"{label} is planned and has not been executed."


__all__ = [
    "ALLOWED_TRANSITIONS",
    "ActionCorrelation",
    "ActionLifecycle",
    "ActionLifecycleSnapshot",
    "ActionState",
    "ActionTransition",
    "InvalidActionTransition",
    "lifecycle_from_legacy_status",
    "render_action_outcome",
]
