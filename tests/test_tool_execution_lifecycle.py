from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rex.tools.dispatcher import ToolDispatcher
from rex.tools.execution import ToolExecutionLifecycle, ToolOutcome
from rex.tools.registry import Tool, ToolRegistry


@dataclass
class FakeTool:
    name: str = "change_thing"
    handler: Any = lambda **_kwargs: {"ok": True}
    operation: str = "mutation"
    risk: str = "safe"
    requires_identity: bool = True
    required_args: tuple[str, ...] = ("value",)
    verifier: Any = None


@pytest.fixture(autouse=True)
def no_disk_audit():
    logger = MagicMock()
    with patch("rex.tools.execution.get_audit_logger", return_value=logger):
        yield logger


def execute(tool: FakeTool, args=None, context=None, timeout=1.0):
    return ToolExecutionLifecycle().execute(
        tool,
        {"value": 1} if args is None else args,
        (
            {"user_id": "james", "request_id": f"request-{time.time_ns()}"}
            if context is None
            else context
        ),
        timeout_seconds=timeout,
    )


def test_missing_arguments_and_identity_fail_before_handler() -> None:
    handler = MagicMock()
    tool = FakeTool(handler=handler)
    missing = execute(tool, args={}, context={"user_id": "james", "request_id": "missing"})
    invalid = execute(
        tool,
        context={"user_id": "../cole", "request_id": "invalid-identity"},
    )
    assert missing.status == ToolOutcome.DENIED
    assert invalid.status == ToolOutcome.DENIED
    handler.assert_not_called()


def test_permission_and_prohibited_risk_fail_closed() -> None:
    denied = execute(
        FakeTool(),
        context={
            "user_id": "james",
            "request_id": "permission",
            "permitted_users": ["cole"],
        },
    )
    prohibited = execute(FakeTool(risk="prohibited"))
    assert denied.status == ToolOutcome.DENIED
    assert prohibited.status == ToolOutcome.DENIED


def test_sensitive_operation_requires_confirmation() -> None:
    handler = MagicMock()
    result = execute(FakeTool(handler=handler, risk="sensitive"))
    assert result.status == ToolOutcome.CONFIRMATION_REQUIRED
    handler.assert_not_called()


def test_http_style_success_without_verifier_is_not_completion() -> None:
    result = execute(FakeTool(handler=lambda **_kwargs: {"ok": True, "http_status": 200}))
    assert result.status == ToolOutcome.ATTEMPTED_UNVERIFIED
    assert result.success is False
    assert "not independently verified" in (result.detail or "")


def test_timeout_after_possible_write_is_unverified() -> None:
    def slow_handler(**_kwargs):
        time.sleep(0.05)
        return {"ok": True}

    result = execute(FakeTool(handler=slow_handler), timeout=0.001)
    assert result.status == ToolOutcome.ATTEMPTED_UNVERIFIED
    assert "possible write" in (result.error or "")


def test_mutation_transient_failure_is_not_retried() -> None:
    handler = MagicMock(side_effect=ConnectionError("network blip"))
    result = execute(FakeTool(handler=handler))
    assert result.status == ToolOutcome.FAILED
    handler.assert_called_once()


def test_duplicate_request_is_not_dispatched_twice_and_mismatch_is_denied() -> None:
    handler = MagicMock(return_value={"ok": True})
    tool = FakeTool(handler=handler)
    context = {"user_id": "james", "request_id": "dedupe-request"}
    first = execute(tool, {"value": 1}, context)
    second = execute(tool, {"value": 1}, context)
    mismatch = execute(tool, {"value": 2}, context)
    assert second is first
    assert mismatch.status == ToolOutcome.DENIED
    handler.assert_called_once()


def test_independent_verifier_controls_mutation_success() -> None:
    verified = execute(
        FakeTool(
            handler=lambda **_kwargs: {"actual": 7},
            verifier=lambda args, output: output["actual"] == args["value"],
        ),
        {"value": 7},
        {"user_id": "james", "request_id": "verified-request"},
    )
    stale = execute(
        FakeTool(
            handler=lambda **_kwargs: {"actual": 6},
            verifier=lambda args, output: output["actual"] == args["value"],
        ),
        {"value": 7},
        {"user_id": "james", "request_id": "stale-request"},
    )
    assert verified.status == ToolOutcome.VERIFIED
    assert verified.success is True
    assert stale.status == ToolOutcome.ATTEMPTED_UNVERIFIED


def test_read_only_tool_completes_without_mutation_language(no_disk_audit) -> None:
    tool = FakeTool(
        name="read_thing",
        handler=lambda **_kwargs: {"value": 3},
        operation="read",
        requires_identity=False,
        required_args=(),
    )
    result = execute(tool, args={}, context={"request_id": "read-request"})
    assert result.status == ToolOutcome.COMPLETED
    assert result.success is True
    assert result.stages == (
        "capability_availability",
        "argument_validation",
        "identity_validation",
        "permission_evaluation",
        "risk_classification",
        "confirmation",
        "execution",
        "normalized_result",
        "independent_verification",
        "truthful_response",
        "audit_recording",
    )
    no_disk_audit.log.assert_called_once()
    audit_entry = no_disk_audit.log.call_args.args[0]
    assert audit_entry.tool_call_args["argument_names"] == []
    assert "value" not in audit_entry.tool_call_args


def test_handler_normalized_outcome_is_preserved() -> None:
    result = execute(
        FakeTool(
            handler=lambda **_kwargs: {
                "status": "confirmation_required",
                "detail": "Confirm lock.unlock.",
            }
        )
    )
    assert result.status == ToolOutcome.CONFIRMATION_REQUIRED
    assert result.detail == "Confirm lock.unlock."


def test_canonical_dispatcher_enforces_lifecycle_for_mutations() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="http_mutation",
            description="Test mutation",
            capability_tags=[],
            requires_config=[],
            handler=lambda target: {"ok": True, "http_status": 200, "target": target},
            operation="mutation",
            requires_identity=True,
            required_args=("target",),
        )
    )
    result = ToolDispatcher(registry).dispatch(
        "http_mutation",
        {"target": "device-1"},
        {"user_id": "james", "request_id": "dispatcher-mutation"},
    )
    assert result.status == ToolOutcome.ATTEMPTED_UNVERIFIED
    assert result.success is False
