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
from typing import Any

from rex.contracts import ToolCall
from rex.openclaw.tool_executor import ApprovalRequiredError, PolicyDeniedError
from rex.policy import PolicyDecision
from rex.policy_engine import PolicyEngine, get_policy_engine

logger = logging.getLogger(__name__)


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
            :exc:`~rex.openclaw.tool_executor.PolicyDeniedError`: When the policy
                explicitly denies the tool call.
            :exc:`~rex.openclaw.tool_executor.ApprovalRequiredError`: When the policy
                requires user approval before execution.
        """
        decision = self.check(tool_name, metadata)
        if decision.denied:
            raise PolicyDeniedError(tool_name, decision.reason)
        if decision.requires_approval:
            raise ApprovalRequiredError(tool_name, decision.reason)

    # ------------------------------------------------------------------
    # Backward-compatible OpenClaw registration shim
    # ------------------------------------------------------------------

    def register(self, agent: object | None = None) -> None:
        """No-op registration shim for older integration tests/call-sites.

        Historical OpenClaw prototypes called ``PolicyAdapter.register()``
        during setup. The current HTTP-first architecture performs policy
        checks directly in Rex and does not require explicit registration.

        Args:
            agent: Accepted for API compatibility with older call-sites.

        Returns:
            Always returns ``None``.
        """
        del agent
        logger.debug("PolicyAdapter.register() is a no-op in HTTP mode.")
        return None
