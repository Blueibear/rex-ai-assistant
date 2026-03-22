"""OpenClaw policy adapter — US-P3-008.

Wraps Rex's :class:`~rex.policy_engine.PolicyEngine` as an OpenClaw
pre-execution hook so that Rex's risk/approval rules apply to every tool
call dispatched by OpenClaw.

When the ``openclaw`` package is not installed, :func:`register` logs a
warning and returns ``None``.  The :meth:`PolicyAdapter.check` and
:meth:`PolicyAdapter.guard` methods work without OpenClaw installed.

Typical usage::

    from rex.openclaw.policy_adapter import PolicyAdapter

    adapter = PolicyAdapter()

    # Low-level check — returns a PolicyDecision
    decision = adapter.check("time_now")
    print(decision.allowed, decision.requires_approval)

    # Guard — raises on deny or approval-required
    adapter.guard("send_email", metadata={"recipient": "user@example.com"})
"""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import Any

from rex.contracts import ToolCall
from rex.policy import PolicyDecision
from rex.policy_engine import PolicyEngine, get_policy_engine
from rex.tool_router import ApprovalRequiredError, PolicyDeniedError

logger = logging.getLogger(__name__)

OPENCLAW_AVAILABLE: bool = find_spec("openclaw") is not None

if OPENCLAW_AVAILABLE:  # pragma: no cover
    import openclaw as _openclaw  # type: ignore[import-not-found]
else:
    _openclaw = None  # type: ignore[assignment]


class PolicyAdapter:
    """Adapter that presents Rex's PolicyEngine as an OpenClaw hook.

    When ``openclaw`` is installed, :meth:`register` registers the adapter
    as a pre-execution hook so that OpenClaw calls :meth:`guard` before
    any tool is dispatched (stub — filled in once the OpenClaw hook API is
    confirmed, see PRD §8.3).

    Without OpenClaw the adapter is still useful as a thin convenience
    wrapper around :class:`~rex.policy_engine.PolicyEngine`.

    Args:
        engine: Optional :class:`~rex.policy_engine.PolicyEngine` instance
            to delegate to.  When *None*, the module-level singleton
            returned by :func:`~rex.policy_engine.get_policy_engine` is
            used.  Inject a custom instance in tests to control policies
            without touching the global singleton.
    """

    def __init__(self, engine: PolicyEngine | None = None) -> None:
        self._engine = engine or get_policy_engine()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def engine(self) -> PolicyEngine:
        """The underlying :class:`~rex.policy_engine.PolicyEngine`."""
        return self._engine

    # ------------------------------------------------------------------
    # Core evaluation methods
    # ------------------------------------------------------------------

    def check(
        self,
        tool_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        """Evaluate Rex's policy for *tool_name* and return the decision.

        Delegates to :meth:`~rex.policy_engine.PolicyEngine.decide` on the
        wrapped engine.  Does **not** raise; callers inspect the returned
        :class:`~rex.policy.PolicyDecision` themselves.

        Args:
            tool_name: Name of the tool to evaluate.
            metadata: Optional context for the decision (e.g.
                ``{"recipient": "user@example.com"}``).

        Returns:
            A :class:`~rex.policy.PolicyDecision` with ``allowed``,
            ``requires_approval``, and ``denied`` fields.
        """
        tool_call = ToolCall(tool=tool_name, args={})
        return self._engine.decide(tool_call, metadata or {})

    def guard(
        self,
        tool_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Raise if Rex's policy blocks or requires approval for *tool_name*.

        Intended as a lightweight pre-flight check that can be inserted
        into any tool-dispatch path.

        Args:
            tool_name: Name of the tool to guard.
            metadata: Optional context forwarded to the policy engine.

        Raises:
            :exc:`~rex.tool_router.PolicyDeniedError`: When the policy
                explicitly denies the tool call.
            :exc:`~rex.tool_router.ApprovalRequiredError`: When the policy
                requires user approval before execution.
        """
        decision = self.check(tool_name, metadata)
        if decision.denied:
            raise PolicyDeniedError(tool_name, decision.reason)
        if decision.requires_approval:
            raise ApprovalRequiredError(tool_name, decision.reason)

    # ------------------------------------------------------------------
    # OpenClaw registration
    # ------------------------------------------------------------------

    def register(self, agent: Any = None) -> Any:
        """Register this adapter as an OpenClaw pre-execution hook.

        When ``openclaw`` is installed, this method registers
        :meth:`guard` so that OpenClaw calls it before dispatching any
        tool.  When OpenClaw is absent, logs a warning and returns
        ``None``.

        .. note::
            The exact OpenClaw hook registration call is a stub (see PRD
            §8.3 — *"Confirm OpenClaw's hook/middleware registration
            API"*).  Replace the ``# TODO`` below once the API is
            confirmed.

        Args:
            agent: Optional OpenClaw agent handle.

        Returns:
            The hook registration handle from OpenClaw, or ``None``.
        """
        if not OPENCLAW_AVAILABLE:
            logger.warning("openclaw package not installed — PolicyAdapter not registered as hook")
            return None

        # TODO: replace with real OpenClaw hook registration once API is confirmed.
        # Expected shape (to be verified):
        #   handle = _openclaw.register_hook(
        #       event="before_tool_call",
        #       handler=self.guard,
        #       agent=agent,
        #   )
        #   return handle
        logger.warning(
            "OpenClaw policy hook registration stub — update once API is confirmed (PRD §8.3)"
        )
        return None
