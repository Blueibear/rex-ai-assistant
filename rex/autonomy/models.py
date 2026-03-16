"""Planning data structures and planner protocol for the Rex autonomy engine.

Defines:
- ``PlanStep``      — a single executable step inside a plan.
- ``Plan``          — an ordered collection of steps with lifecycle tracking.
- ``PlannerProtocol`` — abstract base class that every planner must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StepStatus(str, Enum):
    """Lifecycle status of a single plan step."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStatus(str, Enum):
    """Lifecycle status of an overall plan."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    """A single executable step within a :class:`Plan`.

    Each step maps to one tool invocation.  The planner populates ``tool``
    and ``args``; the runner fills in ``status``, ``result``, and ``error``
    as execution proceeds.
    """

    id: str = Field(..., description="Unique identifier for this step")
    tool: str = Field(..., description="Name of the tool to invoke")
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments to pass to the tool",
    )
    description: str = Field(
        ...,
        description="Human-readable explanation of what this step does",
    )
    status: StepStatus = Field(
        default=StepStatus.PENDING,
        description="Current execution status of this step",
    )
    result: str | None = Field(
        default=None,
        description="Output returned by the tool on success",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the step failed",
    )
    tokens_used: int | None = Field(
        default=None,
        description="Number of LLM tokens consumed by this step, if known",
    )
    cost_usd: float | None = Field(
        default=None,
        description="Estimated cost in USD for this step's LLM calls, if known",
    )


class Plan(BaseModel):
    """An ordered sequence of :class:`PlanStep` objects that achieves a goal.

    Created by a :class:`PlannerProtocol` implementation and executed by the
    autonomy runner.
    """

    id: str = Field(..., description="Unique identifier for this plan")
    goal: str = Field(
        ...,
        description="The natural-language goal this plan is intended to achieve",
    )
    steps: list[PlanStep] = Field(
        default_factory=list,
        description="Ordered list of steps to execute",
    )
    status: PlanStatus = Field(
        default=PlanStatus.PENDING,
        description="Current lifecycle status of the plan",
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="UTC timestamp when the plan was created",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="UTC timestamp when the plan completed or failed; None while in progress",
    )

    @property
    def total_cost_usd(self) -> float:
        """Sum of ``cost_usd`` across all steps that have a known cost."""
        return sum(s.cost_usd for s in self.steps if s.cost_usd is not None)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class PlannerProtocol(ABC):
    """Abstract base class for planner implementations.

    Subclasses must implement :meth:`plan`, which translates a natural-language
    goal into an executable :class:`Plan`.
    """

    @abstractmethod
    def plan(self, goal: str, context: dict[str, Any]) -> Plan:
        """Convert a goal into an executable plan.

        Args:
            goal: Natural-language description of what the user wants to achieve.
            context: Arbitrary key/value context (available tools, user prefs, etc.).

        Returns:
            A :class:`Plan` with one or more :class:`PlanStep` objects.
        """
        ...


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "StepStatus",
    "PlanStatus",
    "PlanStep",
    "Plan",
    "PlannerProtocol",
]
