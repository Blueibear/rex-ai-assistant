"""Bounded fail-closed execution for canonical action dependency graphs."""

from __future__ import annotations

import concurrent.futures
import contextvars
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rex.actions.graph import ActionGraph, ActionNode
from rex.actions.lifecycle import (
    ActionLifecycle,
    ActionLifecycleSnapshot,
    ActionState,
    lifecycle_from_legacy_status,
)
from rex.runtime.cancellation import TurnCancelledError, current_turn_cancellation
from rex.tools.execution import ToolOperation
from rex.tools.protocol import ToolDispatcherProtocol, ToolResult


@dataclass(frozen=True)
class ActionExecutionRecord:
    """Truthful scheduler result for one action node."""

    node: ActionNode
    lifecycle: ActionLifecycleSnapshot
    tool_result: ToolResult | None = None
    blocked_reason: str | None = None

    @property
    def success(self) -> bool:
        if self.blocked_reason is not None:
            return False
        if self.node.operation is ToolOperation.MUTATION:
            return self.lifecycle.state is ActionState.VERIFIED
        return self.lifecycle.success


@dataclass(frozen=True)
class ActionGraphResult:
    plan_id: str
    records: dict[str, ActionExecutionRecord]

    @property
    def success(self) -> bool:
        return bool(self.records) and all(record.success for record in self.records.values())


class ActionGraphExecutor:
    """Execute a validated graph without bypassing canonical tool authority."""

    def __init__(
        self,
        dispatcher: ToolDispatcherProtocol,
        *,
        operation_resolver: Callable[[str], ToolOperation | str],
        confirmation_resolver: Callable[[ActionNode, dict[str, Any]], bool] | None = None,
        max_parallel_reads: int = 4,
    ) -> None:
        if max_parallel_reads < 1:
            raise ValueError("max_parallel_reads must be at least 1")
        self._dispatcher = dispatcher
        self._operation_resolver = operation_resolver
        self._confirmation_resolver = confirmation_resolver
        self._max_parallel_reads = max_parallel_reads

    def execute(
        self,
        graph: ActionGraph,
        context: dict[str, Any] | None = None,
    ) -> ActionGraphResult:
        ambient = dict(context or {})
        records: dict[str, ActionExecutionRecord] = {}
        confirmations: dict[str, bool] = {}
        pending = list(graph.nodes)

        while pending:
            cancellation = current_turn_cancellation()
            if cancellation is not None and cancellation.cancelled:
                for node in pending:
                    records[node.action_id] = self._cancelled_record(graph, node)
                break

            self._block_failed_descendants(graph, pending, records)
            pending = [node for node in pending if node.action_id not in records]
            if not pending:
                break

            ready = self._ready_nodes(graph, pending, records)
            if not ready:
                raise RuntimeError("validated action graph made no scheduling progress")

            for node in tuple(ready):
                confirmed = (
                    self._is_confirmed(node, ambient) if node.confirmation_required else False
                )
                confirmations[node.action_id] = confirmed
                reason = self._boundary_block_reason(node, confirmed)
                if reason is None:
                    reason = self._operation_block_reason(node)
                if reason is not None:
                    records[node.action_id] = self._blocked_record(graph, node, reason)

            pending = [node for node in pending if node.action_id not in records]
            ready = [node for node in ready if node.action_id not in records]
            if not ready:
                continue

            batch = self._next_batch(ready)
            batch_records = self._execute_batch(graph, batch, ambient, confirmations)
            records.update(batch_records)
            pending = [node for node in pending if node.action_id not in records]

        return ActionGraphResult(plan_id=graph.plan_id, records=records)

    @staticmethod
    def _ready_nodes(
        graph: ActionGraph,
        pending: list[ActionNode],
        records: dict[str, ActionExecutionRecord],
    ) -> list[ActionNode]:
        del graph
        return [
            node
            for node in pending
            if all(dependency in records for dependency in node.dependencies)
            and all(records[dependency].success for dependency in node.dependencies)
        ]

    @classmethod
    def _block_failed_descendants(
        cls,
        graph: ActionGraph,
        pending: list[ActionNode],
        records: dict[str, ActionExecutionRecord],
    ) -> None:
        for node in pending:
            failed_dependencies = [
                dependency
                for dependency in node.dependencies
                if dependency in records and not records[dependency].success
            ]
            if failed_dependencies:
                reason = "Blocked by unsuccessful dependency: " + ", ".join(failed_dependencies)
                records[node.action_id] = cls._blocked_record(graph, node, reason)

    @staticmethod
    def _boundary_block_reason(node: ActionNode, confirmed: bool) -> str | None:
        if not node.authorized:
            return "Authorization is required before scheduling this action"
        if node.confirmation_required and not confirmed:
            return "Confirmation is required before scheduling this action"
        return None

    def _is_confirmed(self, node: ActionNode, ambient: dict[str, Any]) -> bool:
        if self._confirmation_resolver is None:
            return False
        try:
            return bool(self._confirmation_resolver(node, ambient))
        except Exception:
            return False

    def _operation_block_reason(self, node: ActionNode) -> str | None:
        try:
            canonical = ToolOperation(self._operation_resolver(node.tool_name))
        except Exception:
            return "Canonical tool operation metadata is unavailable"
        if canonical is not node.operation:
            return "Declared operation does not match canonical tool operation metadata"
        return None

    def _next_batch(self, ready: list[ActionNode]) -> list[ActionNode]:
        first = ready[0]
        if first.operation is ToolOperation.MUTATION:
            return [first]

        batch: list[ActionNode] = []
        claimed: set[str] = set()
        for node in ready:
            if node.operation is ToolOperation.MUTATION or len(batch) >= self._max_parallel_reads:
                break
            resources = set(node.conflict_keys) or {"*"}
            conflicts = "*" in resources or "*" in claimed or bool(resources & claimed)
            if batch and conflicts:
                break
            batch.append(node)
            claimed.update(resources)
        return batch or [first]

    def _execute_batch(
        self,
        graph: ActionGraph,
        batch: list[ActionNode],
        ambient: dict[str, Any],
        confirmations: dict[str, bool],
    ) -> dict[str, ActionExecutionRecord]:
        if len(batch) == 1:
            node = batch[0]
            return {
                node.action_id: self._execute_node(
                    graph, node, ambient, confirmations.get(node.action_id, False)
                )
            }

        records: dict[str, ActionExecutionRecord] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self._max_parallel_reads, len(batch))
        ) as pool:
            futures = {
                pool.submit(
                    contextvars.copy_context().run,
                    self._execute_node,
                    graph,
                    node,
                    ambient,
                    confirmations.get(node.action_id, False),
                ): node
                for node in batch
            }
            for future, node in futures.items():
                records[node.action_id] = future.result()
        return records

    def _execute_node(
        self,
        graph: ActionGraph,
        node: ActionNode,
        ambient: dict[str, Any],
        confirmed: bool,
    ) -> ActionExecutionRecord:
        context = {
            **ambient,
            "request_id": node.action_id,
            "plan_id": graph.plan_id,
            "confirmed": confirmed,
        }
        try:
            result = self._dispatcher.dispatch(node.tool_name, dict(node.args), context)
        except TurnCancelledError:
            return self._cancelled_record(graph, node)
        except Exception as exc:
            return self._failed_record(graph, node, str(exc))

        lifecycle = result.lifecycle or lifecycle_from_legacy_status(
            result.status,
            action_id=node.action_id,
            plan_id=graph.plan_id,
        )
        if lifecycle.state is ActionState.COMPLETED and (
            node.operation is ToolOperation.MUTATION or node.verification_required
        ):
            lifecycle = lifecycle_from_legacy_status(
                "unverified",
                action_id=node.action_id,
                plan_id=graph.plan_id,
            )
        return ActionExecutionRecord(node=node, lifecycle=lifecycle, tool_result=result)

    @staticmethod
    def _blocked_record(
        graph: ActionGraph,
        node: ActionNode,
        reason: str,
    ) -> ActionExecutionRecord:
        lifecycle = ActionLifecycle.create(
            action_id=node.action_id,
            plan_id=graph.plan_id,
        ).snapshot()
        return ActionExecutionRecord(node=node, lifecycle=lifecycle, blocked_reason=reason)

    @staticmethod
    def _cancelled_record(graph: ActionGraph, node: ActionNode) -> ActionExecutionRecord:
        lifecycle = ActionLifecycle.create(
            action_id=node.action_id,
            plan_id=graph.plan_id,
        ).transition(ActionState.CANCELLED, evidence_ref="scheduler:cancelled")
        return ActionExecutionRecord(node=node, lifecycle=lifecycle)

    @staticmethod
    def _failed_record(
        graph: ActionGraph,
        node: ActionNode,
        detail: str,
    ) -> ActionExecutionRecord:
        lifecycle = ActionLifecycle.create(
            action_id=node.action_id,
            plan_id=graph.plan_id,
        ).transition(ActionState.FAILED, evidence_ref="scheduler:dispatch_failed")
        return ActionExecutionRecord(
            node=node,
            lifecycle=lifecycle,
            blocked_reason=detail,
        )


__all__ = [
    "ActionExecutionRecord",
    "ActionGraphExecutor",
    "ActionGraphResult",
]
