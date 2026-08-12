from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from rex.actions.graph import ActionGraph, ActionNode
from rex.actions.graph_executor import ActionGraphExecutor
from rex.actions.lifecycle import ActionState, lifecycle_from_legacy_status
from rex.runtime.cancellation import (
    TurnCancellation,
    current_turn_cancellation,
    turn_cancellation_scope,
)
from rex.tools.dispatcher import ToolDispatcher
from rex.tools.execution import ToolOperation
from rex.tools.protocol import ToolResult
from rex.tools.registry import Tool, ToolRegistry


def _result(action_id: str, status: str, *, plan_id: str = "plan") -> ToolResult:
    lifecycle = lifecycle_from_legacy_status(status, action_id=action_id, plan_id=plan_id)
    return ToolResult(
        success=lifecycle.success,
        status=status,
        request_id=action_id,
        lifecycle=lifecycle,
    )


@dataclass
class RecordingDispatcher:
    statuses: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0
    calls: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    max_active: int = 0
    _active: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def dispatch(
        self,
        name: str,
        args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> ToolResult:
        action_id = str((context or {})["request_id"])
        plan_id = str((context or {})["plan_id"])
        with self._lock:
            self.calls.append(action_id)
            self.events.append(f"start:{action_id}")
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if self.delay:
                time.sleep(self.delay)
            status = self.statuses.get(action_id, "completed")
            return _result(action_id, status, plan_id=plan_id)
        finally:
            with self._lock:
                self._active -= 1
                self.events.append(f"end:{action_id}")


def _executor(
    dispatcher: RecordingDispatcher,
    operations: dict[str, ToolOperation],
    *,
    max_parallel_reads: int = 2,
) -> ActionGraphExecutor:
    return ActionGraphExecutor(
        dispatcher,
        operation_resolver=lambda name: operations[name],
        max_parallel_reads=max_parallel_reads,
    )


def test_independent_reads_run_concurrently_with_bounded_parallelism() -> None:
    dispatcher = RecordingDispatcher(delay=0.20)
    operations = dict.fromkeys(("r1", "r2", "r3"), ToolOperation.READ)
    graph = ActionGraph(
        plan_id="parallel-reads",
        nodes=tuple(
            ActionNode(name, name, conflict_keys=(f"resource:{name}",)) for name in operations
        ),
    )
    started = time.monotonic()

    result = _executor(dispatcher, operations).execute(graph, {"user_id": "james"})
    elapsed = time.monotonic() - started

    assert result.success is True
    assert dispatcher.max_active == 2
    assert elapsed < 0.55


def test_conflicting_reads_and_mutations_serialize_deterministically() -> None:
    dispatcher = RecordingDispatcher(
        statuses={"mutate-a": "verified", "mutate-b": "verified"},
        delay=0.02,
    )
    operations = {
        "read-a": ToolOperation.READ,
        "read-b": ToolOperation.READ,
        "mutate-a": ToolOperation.MUTATION,
        "mutate-b": ToolOperation.MUTATION,
    }
    graph = ActionGraph(
        plan_id="serialize",
        nodes=(
            ActionNode("read-a", "read-a", conflict_keys=("shared",)),
            ActionNode("read-b", "read-b", conflict_keys=("shared",)),
            ActionNode("mutate-a", "mutate-a", operation=ToolOperation.MUTATION),
            ActionNode("mutate-b", "mutate-b", operation=ToolOperation.MUTATION),
        ),
    )

    result = _executor(dispatcher, operations).execute(graph)

    assert result.success is True
    assert dispatcher.max_active == 1
    assert dispatcher.events == [
        "start:read-a",
        "end:read-a",
        "start:read-b",
        "end:read-b",
        "start:mutate-a",
        "end:mutate-a",
        "start:mutate-b",
        "end:mutate-b",
    ]


@pytest.mark.parametrize("terminal_status", ["failed", "cancelled", "unverified"])
def test_unsuccessful_dependency_blocks_descendants_without_dispatch(
    terminal_status: str,
) -> None:
    dispatcher = RecordingDispatcher(statuses={"root": terminal_status})
    operations = {"root": ToolOperation.READ, "child": ToolOperation.READ}
    graph = ActionGraph(
        plan_id="blocked-descendant",
        nodes=(
            ActionNode("root", "root"),
            ActionNode("child", "child", dependencies=("root",)),
        ),
    )

    result = _executor(dispatcher, operations).execute(graph)

    assert dispatcher.calls == ["root"]
    assert result.records["child"].lifecycle.state is ActionState.PLANNED
    assert result.records["child"].blocked_reason is not None
    assert "dependency" in result.records["child"].blocked_reason.lower()
    assert result.success is False


def test_confirmation_and_authorization_boundaries_block_before_dispatch() -> None:
    dispatcher = RecordingDispatcher()
    operations = {"needs-confirmation": ToolOperation.MUTATION, "denied": ToolOperation.READ}
    graph = ActionGraph(
        plan_id="boundaries",
        nodes=(
            ActionNode(
                "needs-confirmation",
                "needs-confirmation",
                operation=ToolOperation.MUTATION,
                confirmation_required=True,
            ),
            ActionNode("denied", "denied", authorized=False),
        ),
    )

    result = _executor(dispatcher, operations).execute(graph)

    assert dispatcher.calls == []
    assert result.records["needs-confirmation"].lifecycle.state is ActionState.PLANNED
    assert "confirmation" in result.records["needs-confirmation"].blocked_reason.lower()
    assert result.records["denied"].lifecycle.state is ActionState.PLANNED
    assert "authorization" in result.records["denied"].blocked_reason.lower()


def test_declared_operation_must_match_canonical_metadata() -> None:
    dispatcher = RecordingDispatcher()
    graph = ActionGraph(plan_id="mismatch", nodes=(ActionNode("a", "tool-a"),))
    result = _executor(dispatcher, {"tool-a": ToolOperation.MUTATION}).execute(graph)

    assert dispatcher.calls == []
    assert result.records["a"].lifecycle.state is ActionState.PLANNED
    assert "operation" in result.records["a"].blocked_reason.lower()


def test_verification_required_mutation_fails_closed_on_completed_only_result() -> None:
    dispatcher = RecordingDispatcher(statuses={"mutate": "completed"})
    graph = ActionGraph(
        plan_id="verification-boundary",
        nodes=(
            ActionNode(
                "mutate",
                "mutate",
                operation=ToolOperation.MUTATION,
                verification_required=True,
                postcondition="resource changed",
            ),
        ),
    )

    result = _executor(dispatcher, {"mutate": ToolOperation.MUTATION}).execute(graph)

    record = result.records["mutate"]
    assert record.lifecycle.state is ActionState.UNVERIFIED
    assert record.success is False
    assert result.success is False


def test_real_dispatcher_rechecks_confirmation_and_permissions(monkeypatch) -> None:
    monkeypatch.setattr(
        "rex.tools.execution.get_audit_logger",
        lambda: SimpleNamespace(log=lambda _entry: None),
    )
    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="sensitive_write",
            description="Mutate a sensitive resource",
            capability_tags=["write"],
            requires_config=[],
            handler=lambda **_kwargs: calls.append("called") or {"ok": True},
            operation="mutation",
            risk="sensitive",
            requires_identity=True,
            required_permissions=("resource_write",),
            verifier=lambda _args, _output: True,
        )
    )
    dispatcher = ToolDispatcher(registry)
    executor = ActionGraphExecutor(
        dispatcher,
        operation_resolver=lambda name: ToolOperation(registry.get(name).operation),
    )

    def run(action_id: str, *, permissions: set[str]):
        graph = ActionGraph(
            plan_id=f"plan-{action_id}",
            nodes=(
                ActionNode(
                    action_id,
                    "sensitive_write",
                    operation=ToolOperation.MUTATION,
                    verification_required=True,
                ),
            ),
        )
        return executor.execute(
            graph,
            {
                "user_id": "james",
                "granted_permissions": permissions,
                "confirmed": True,
            },
        ).records[action_id]

    confirmation = run("confirm-live-policy", permissions={"resource_write"})
    denied = run("permission-live-policy", permissions=set())

    assert calls == []
    assert confirmation.lifecycle.state is ActionState.PLANNED
    assert confirmation.tool_result is not None
    assert confirmation.tool_result.status == "confirmation_required"
    assert denied.lifecycle.state is ActionState.FAILED
    assert denied.tool_result is not None
    assert denied.tool_result.status == "denied"


def test_only_trusted_confirmation_resolver_can_release_graph_boundary() -> None:
    dispatcher = RecordingDispatcher(statuses={"mutate": "verified"})
    graph = ActionGraph(
        plan_id="trusted-confirmation",
        nodes=(
            ActionNode(
                "mutate",
                "mutate",
                operation=ToolOperation.MUTATION,
                confirmation_required=True,
                verification_required=True,
            ),
        ),
    )
    resolution_calls: list[str] = []

    def resolve_confirmation(node: ActionNode, context: dict[str, Any]) -> bool:
        resolution_calls.append(node.action_id)
        return context.get("approved_action") == node.action_id

    executor = ActionGraphExecutor(
        dispatcher,
        operation_resolver=lambda _name: ToolOperation.MUTATION,
        confirmation_resolver=resolve_confirmation,
    )

    blocked = executor.execute(graph).records["mutate"]
    allowed = executor.execute(graph, {"approved_action": "mutate"}).records["mutate"]

    assert blocked.lifecycle.state is ActionState.PLANNED
    assert blocked.blocked_reason is not None
    assert dispatcher.calls == ["mutate"]
    assert allowed.lifecycle.state is ActionState.VERIFIED
    assert resolution_calls == ["mutate", "mutate"]


class CancellationAwareDispatcher:
    def __init__(self, cancellation: TurnCancellation) -> None:
        self.cancellation = cancellation
        self.calls: list[str] = []
        self._lock = threading.Lock()
        self._started = threading.Event()
        self._active = 0

    def dispatch(
        self,
        name: str,
        args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> ToolResult:
        del name, args, context
        bound = current_turn_cancellation()
        assert bound is self.cancellation
        with self._lock:
            self.calls.append(threading.current_thread().name)
            self._active += 1
            if self._active == 2:
                self._started.set()
        assert self._started.wait(timeout=1.0)
        self.cancellation.cancel("parallel-test")
        bound.raise_if_cancelled()
        raise AssertionError("cancellation should have raised")


def test_turn_cancellation_propagates_into_parallel_workers_and_stops_descendant() -> None:
    cancellation = TurnCancellation("parallel-turn")
    dispatcher = CancellationAwareDispatcher(cancellation)
    operations = dict.fromkeys(("root-a", "root-b", "child"), ToolOperation.READ)
    graph = ActionGraph(
        plan_id="cancel-parallel",
        nodes=(
            ActionNode("root-a", "root-a", conflict_keys=("resource:a",)),
            ActionNode("root-b", "root-b", conflict_keys=("resource:b",)),
            ActionNode(
                "child",
                "child",
                dependencies=("root-a", "root-b"),
                conflict_keys=("resource:child",),
            ),
        ),
    )

    with turn_cancellation_scope(cancellation):
        result = _executor(dispatcher, operations).execute(graph)

    assert len(dispatcher.calls) == 2
    assert result.records["root-a"].lifecycle.state is ActionState.CANCELLED
    assert result.records["root-b"].lifecycle.state is ActionState.CANCELLED
    assert result.records["child"].lifecycle.state is ActionState.CANCELLED
    assert result.success is False
