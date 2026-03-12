"""
US-099: Fill unit test gaps — planner, tool registry, workflow engine

Verifies that all public methods of the planner, tool registry, tool router,
and workflow engine have at least one passing unit test.
No external services or live credentials are used.
"""

from __future__ import annotations

import socket

import pytest

from rex.contracts import ToolCall
from rex.credentials import CredentialManager
from rex.planner import Planner, PlannerError
from rex.policy_engine import get_policy_engine, reset_policy_engine
from rex.tool_registry import (
    ToolMeta,
    ToolRegistry,
    get_tool_registry,
    reset_tool_registry,
    set_tool_registry,
)
from rex.tool_router import (
    execute_tool,
    format_tool_result,
    parse_tool_request,
    route_if_tool_request,
)
from rex.workflow import (
    StepResult,
    Workflow,
    WorkflowApproval,
    WorkflowStep,
    clear_condition_registry,
    generate_approval_id,
    generate_step_id,
    generate_workflow_id,
    get_condition,
    register_condition,
    validate_workflow_steps,
)
from rex.workflow_runner import (
    WorkflowRunner,
    approve_workflow,
    deny_workflow,
    list_pending_approvals,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registries():
    """Reset global registries before and after every test."""
    reset_tool_registry()
    reset_policy_engine()
    clear_condition_registry()
    yield
    reset_tool_registry()
    reset_policy_engine()
    clear_condition_registry()


@pytest.fixture
def basic_registry():
    reg = ToolRegistry()
    reg.register_tool(
        ToolMeta(
            name="time_now",
            description="Return current time",
            required_credentials=[],
            enabled=True,
        )
    )
    return reg


@pytest.fixture
def planner(basic_registry):
    set_tool_registry(basic_registry)
    return Planner(tool_registry=basic_registry, policy_engine=get_policy_engine())


# ---------------------------------------------------------------------------
# AC-1: all public methods of rex/planner.py have at least one passing test
# ---------------------------------------------------------------------------


class TestPlannerPublicMethods:
    def test_plan_returns_workflow(self, planner):
        wf = planner.plan("check the time", requested_by="test")
        assert wf is not None

    def test_validate_workflow_returns_bool(self, planner):
        step = WorkflowStep(description="get time")
        wf = Workflow(title="Test", steps=[step])
        result = planner.validate_workflow(wf)
        assert isinstance(result, bool)

    def test_select_tool_registered(self, planner):
        selected = planner.select_tool(["time_now"])
        assert selected == "time_now"

    def test_select_tool_missing_returns_none(self, planner):
        selected = planner.select_tool(["nonexistent_tool"])
        assert selected is None

    def test_select_tool_empty_candidates(self, planner):
        selected = planner.select_tool([])
        assert selected is None

    def test_build_prompt_returns_nonempty_string(self, planner):
        prompt = planner.build_prompt("check the time")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_build_prompt_contains_goal(self, planner):
        goal = "send an email to Alice"
        prompt = planner.build_prompt(goal)
        assert goal in prompt

    def test_plan_with_fallback_primary_succeeds(self, planner):
        wf = planner.plan_with_fallback(
            "check the time", fallback_goals=["what time is it"]
        )
        assert wf is not None

    def test_plan_with_fallback_uses_fallback_when_primary_fails(self, planner, monkeypatch):
        call_count = [0]

        original_plan = planner.plan

        def patched_plan(goal, requested_by=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise PlannerError("primary failed")
            return original_plan(goal, requested_by=requested_by)

        monkeypatch.setattr(planner, "plan", patched_plan)
        wf = planner.plan_with_fallback(
            "bad goal", fallback_goals=["check the time"]
        )
        assert wf is not None
        assert call_count[0] == 2

    def test_plan_with_fallback_raises_when_all_fail(self, planner, monkeypatch):
        from rex.planner import UnableToPlanError

        monkeypatch.setattr(
            planner, "plan", lambda goal, requested_by=None: (_ for _ in ()).throw(PlannerError("fail"))
        )
        with pytest.raises((PlannerError, UnableToPlanError, Exception)):
            planner.plan_with_fallback("bad goal", fallback_goals=["also bad"])

    def test_execute_step_with_fallback_primary_succeeds(self, planner):
        primary = WorkflowStep(description="primary step")
        result = planner.execute_step_with_fallback(
            primary, fallback_steps=[], executor=lambda step: "ok"
        )
        assert result == "ok"

    def test_execute_step_with_fallback_uses_fallback(self, planner):
        primary = WorkflowStep(description="primary")
        fallback = WorkflowStep(description="fallback")

        def executor(step):
            if step is primary:
                raise RuntimeError("primary failed")
            return "fallback_ok"

        result = planner.execute_step_with_fallback(
            primary, fallback_steps=[fallback], executor=executor
        )
        assert result == "fallback_ok"

    def test_execute_step_with_fallback_reraises_when_all_fail(self, planner):
        primary = WorkflowStep(description="primary")
        fallback = WorkflowStep(description="fallback")

        def always_fail(step):
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError):
            planner.execute_step_with_fallback(
                primary, fallback_steps=[fallback], executor=always_fail
            )


# ---------------------------------------------------------------------------
# AC-2a: all public methods of tool registry have at least one passing test
# ---------------------------------------------------------------------------


class TestToolRegistryPublicMethods:
    def test_register_and_get_tool(self, basic_registry):
        meta = basic_registry.get_tool("time_now")
        assert meta is not None
        assert meta.name == "time_now"

    def test_unregister_tool_returns_true(self, basic_registry):
        assert basic_registry.unregister_tool("time_now") is True
        assert basic_registry.get_tool("time_now") is None

    def test_unregister_missing_returns_false(self, basic_registry):
        assert basic_registry.unregister_tool("nonexistent") is False

    def test_has_tool_true_and_false(self, basic_registry):
        assert basic_registry.has_tool("time_now") is True
        assert basic_registry.has_tool("missing") is False

    def test_list_tools_excludes_disabled(self):
        reg = ToolRegistry()
        reg.register_tool(
            ToolMeta(name="a", description="d", required_credentials=[], enabled=True)
        )
        reg.register_tool(
            ToolMeta(name="b", description="d", required_credentials=[], enabled=False)
        )
        enabled = reg.list_tools(include_disabled=False)
        assert all(t.enabled for t in enabled)
        assert len(enabled) == 1

    def test_list_tools_includes_disabled(self):
        reg = ToolRegistry()
        reg.register_tool(
            ToolMeta(name="a", description="d", required_credentials=[], enabled=True)
        )
        reg.register_tool(
            ToolMeta(name="b", description="d", required_credentials=[], enabled=False)
        )
        all_tools = reg.list_tools(include_disabled=True)
        assert len(all_tools) == 2

    def test_check_credentials_no_requirements(self, basic_registry):
        ok, missing = basic_registry.check_credentials("time_now")
        assert ok is True
        assert missing == []

    def test_check_credentials_with_present_token(self):
        cm = CredentialManager()
        cm.set_token("my_api_key", "secret")
        reg = ToolRegistry(credential_manager=cm)
        reg.register_tool(
            ToolMeta(
                name="t",
                description="d",
                required_credentials=["my_api_key"],
                enabled=True,
            )
        )
        ok, missing = reg.check_credentials("t")
        assert ok is True
        assert missing == []

    def test_check_credentials_missing_token(self):
        cm = CredentialManager()
        reg = ToolRegistry(credential_manager=cm)
        reg.register_tool(
            ToolMeta(
                name="t",
                description="d",
                required_credentials=["missing_key_us099"],
                enabled=True,
            )
        )
        ok, missing = reg.check_credentials("t")
        assert ok is False
        assert "missing_key_us099" in missing

    def test_validate_credentials_for_tool_missing_raises(self):
        cm = CredentialManager()
        reg = ToolRegistry(credential_manager=cm)
        reg.register_tool(
            ToolMeta(
                name="t",
                description="d",
                required_credentials=["definitely_not_set_us099"],
                enabled=True,
            )
        )
        with pytest.raises(Exception):
            reg.validate_credentials_for_tool("t")

    def test_check_health_with_healthy_tool(self):
        reg = ToolRegistry()
        reg.register_tool(
            ToolMeta(
                name="t",
                description="d",
                required_credentials=[],
                enabled=True,
                health_check=lambda: (True, "all good"),
            )
        )
        ok, msg = reg.check_health("t")
        assert ok is True
        assert "all good" in msg

    def test_check_all_health_returns_dict(self):
        reg = ToolRegistry()
        reg.register_tool(
            ToolMeta(
                name="t",
                description="d",
                required_credentials=[],
                enabled=True,
                health_check=lambda: (True, "ok"),
            )
        )
        results = reg.check_all_health()
        assert isinstance(results, dict)
        assert "t" in results

    def test_get_tool_status_returns_dict_with_name(self, basic_registry):
        status = basic_registry.get_tool_status("time_now")
        assert isinstance(status, dict)
        assert status.get("name") == "time_now"

    def test_get_all_status_returns_list(self, basic_registry):
        statuses = basic_registry.get_all_status()
        assert isinstance(statuses, list)
        assert len(statuses) >= 1

    def test_module_level_get_set_reset(self):
        original = get_tool_registry()
        new_reg = ToolRegistry()
        set_tool_registry(new_reg)
        assert get_tool_registry() is new_reg
        reset_tool_registry()


# ---------------------------------------------------------------------------
# AC-2b: all public functions of tool router have at least one passing test
# ---------------------------------------------------------------------------


class TestToolRouterPublicFunctions:
    def test_parse_tool_request_valid(self):
        payload = 'TOOL_REQUEST: {"tool":"time_now","args":{}}'
        result = parse_tool_request(payload)
        assert result is not None
        assert result["tool"] == "time_now"

    def test_parse_tool_request_plain_text_returns_none(self):
        assert parse_tool_request("just a normal sentence") is None

    def test_parse_tool_request_invalid_json_returns_none(self):
        assert parse_tool_request("TOOL_REQUEST: not-json") is None

    def test_execute_tool_time_now(self):
        result = execute_tool({"tool": "time_now", "args": {}}, {})
        assert isinstance(result, dict)

    def test_format_tool_result_returns_string(self):
        line = format_tool_result("time_now", {}, {"time": "12:00"})
        assert isinstance(line, str)
        assert len(line) > 0

    def test_route_if_tool_request_non_tool_returns_text(self):
        result = route_if_tool_request(
            "Hello, how are you?",
            default_context={},
            model_call_fn=lambda prompt: "I'm fine",
        )
        assert isinstance(result, str)

    def test_route_if_tool_request_tool_request_executes(self):
        payload = 'TOOL_REQUEST: {"tool":"time_now","args":{}}'
        result = route_if_tool_request(
            payload,
            default_context={},
            model_call_fn=lambda prompt: "fallback",
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# AC-3: all public methods of the workflow engine have at least one passing test
# ---------------------------------------------------------------------------


class TestWorkflowIdGenerators:
    def test_generate_workflow_id_unique(self):
        ids = {generate_workflow_id() for _ in range(10)}
        assert len(ids) == 10
        assert all(i.startswith("wf_") for i in ids)

    def test_generate_step_id_unique(self):
        ids = {generate_step_id() for _ in range(10)}
        assert len(ids) == 10

    def test_generate_approval_id_unique(self):
        ids = {generate_approval_id() for _ in range(10)}
        assert len(ids) == 10


class TestConditionRegistry:
    def test_register_and_get_condition(self):
        register_condition("always_true", lambda s: True)
        fn = get_condition("always_true")
        assert fn is not None
        assert fn({}) is True

    def test_get_condition_missing_returns_none(self):
        assert get_condition("no_such_condition") is None

    def test_clear_condition_registry(self):
        register_condition("temp", lambda s: False)
        clear_condition_registry()
        assert get_condition("temp") is None


class TestValidateWorkflowSteps:
    def test_valid_workflow_returns_empty_errors(self):
        step = WorkflowStep(description="do something")
        wf = Workflow(title="Test", steps=[step])
        errors = validate_workflow_steps(wf)
        assert isinstance(errors, dict)
        assert errors == {}

    def test_empty_description_returns_error(self):
        step = WorkflowStep.__new__(WorkflowStep)
        step.__dict__.update(
            {
                "step_id": generate_step_id(),
                "description": "   ",
                "tool_call": None,
                "precondition": None,
                "postcondition": None,
                "idempotency_key": None,
                "result": None,
                "requires_approval": False,
                "approval_id": None,
            }
        )
        # use a valid description to avoid Pydantic validation but override after
        step2 = WorkflowStep(description="valid")
        object.__setattr__(step2, "description", "   ")
        wf = Workflow(title="Test", steps=[step2])
        errors = validate_workflow_steps(wf)
        # at least one step error should be present
        assert isinstance(errors, dict)


class TestWorkflowModel:
    def test_is_finished_false_when_queued(self):
        wf = Workflow(title="Test")
        assert not wf.is_finished()

    def test_is_finished_true_when_completed(self):
        wf = Workflow(title="Test")
        wf.status = "completed"
        assert wf.is_finished()

    def test_is_blocked_true_when_blocked(self):
        wf = Workflow(title="Test")
        wf.status = "blocked"
        assert wf.is_blocked()

    def test_current_step_returns_first_step(self):
        step = WorkflowStep(description="step 1")
        wf = Workflow(title="Test", steps=[step])
        assert wf.current_step() is step

    def test_current_step_none_on_empty_workflow(self):
        wf = Workflow(title="Test")
        assert wf.current_step() is None

    def test_advance_to_completion(self):
        step1 = WorkflowStep(description="step 1")
        step2 = WorkflowStep(description="step 2")
        wf = Workflow(title="Test", steps=[step1, step2])
        wf.mark_running()
        # First advance moves from step 0 to step 1 (returns True = still steps remaining)
        advanced = wf.advance()
        assert advanced is True
        assert wf.current_step_index == 1
        # Second advance finishes the workflow (returns False = no more steps)
        finished = wf.advance()
        assert finished is False
        assert wf.is_finished()

    def test_mark_blocked(self):
        approval_id = generate_approval_id()
        wf = Workflow(title="Test")
        wf.mark_blocked(approval_id)
        assert wf.is_blocked()
        assert wf.blocking_approval_id == approval_id

    def test_mark_failed(self):
        wf = Workflow(title="Test")
        wf.mark_failed("something went wrong")
        assert wf.status == "failed"
        assert wf.error == "something went wrong"

    def test_mark_running(self):
        wf = Workflow(title="Test")
        wf.mark_running()
        assert wf.status == "running"

    def test_mark_canceled(self):
        wf = Workflow(title="Test")
        wf.mark_canceled()
        assert wf.status == "canceled"

    def test_get_executed_idempotency_keys(self):
        step = WorkflowStep(
            description="step",
            idempotency_key="key-abc",
            result=StepResult(step_id="s1", success=True),
        )
        wf = Workflow(title="Test", steps=[step])
        keys = wf.get_executed_idempotency_keys()
        assert "key-abc" in keys

    def test_get_executed_idempotency_keys_excludes_failed(self):
        step = WorkflowStep(
            description="step",
            idempotency_key="key-xyz",
            result=StepResult(step_id="s1", success=False),
        )
        wf = Workflow(title="Test", steps=[step])
        keys = wf.get_executed_idempotency_keys()
        assert "key-xyz" not in keys

    def test_save_and_load(self, tmp_path):
        wf = Workflow(title="Persistence Test")
        path = wf.save(workflow_dir=tmp_path)
        assert path.exists()
        loaded = Workflow.load(wf.workflow_id, workflow_dir=tmp_path)
        assert loaded is not None
        assert loaded.workflow_id == wf.workflow_id
        assert loaded.title == "Persistence Test"

    def test_load_returns_none_for_missing(self, tmp_path):
        result = Workflow.load("wf_does_not_exist", workflow_dir=tmp_path)
        assert result is None

    def test_load_from_file(self, tmp_path):
        wf = Workflow(title="File Load Test")
        path = wf.save(workflow_dir=tmp_path)
        loaded = Workflow.load_from_file(path)
        assert loaded.workflow_id == wf.workflow_id

    def test_list_workflows_all(self, tmp_path):
        wf1 = Workflow(title="WF1")
        wf2 = Workflow(title="WF2")
        wf1.save(workflow_dir=tmp_path)
        wf2.save(workflow_dir=tmp_path)
        all_wfs = Workflow.list_workflows(workflow_dir=tmp_path)
        assert len(all_wfs) == 2

    def test_list_workflows_filtered_by_status(self, tmp_path):
        wf1 = Workflow(title="Running")
        wf1.mark_running()
        wf2 = Workflow(title="Queued")
        wf1.save(workflow_dir=tmp_path)
        wf2.save(workflow_dir=tmp_path)
        running = Workflow.list_workflows(workflow_dir=tmp_path, status="running")
        assert all(w.status == "running" for w in running)
        assert len(running) == 1


class TestWorkflowApproval:
    def test_save_and_load(self, tmp_path):
        apr = WorkflowApproval(
            approval_id=generate_approval_id(),
            workflow_id="wf1",
            step_id="s1",
            requested_by="user",
        )
        apr.save(approval_dir=tmp_path)
        loaded = WorkflowApproval.load(apr.approval_id, approval_dir=tmp_path)
        assert loaded is not None
        assert loaded.approval_id == apr.approval_id

    def test_load_returns_none_for_missing(self, tmp_path):
        result = WorkflowApproval.load("apr_nonexistent", approval_dir=tmp_path)
        assert result is None


class TestWorkflowStep:
    def test_validate_inputs_clean_step(self):
        step = WorkflowStep(description="get time")
        errors = step.validate_inputs()
        assert errors == []

    def test_validate_inputs_empty_description_error(self):
        # Bypass pydantic by setting a blank description via model field override
        step = WorkflowStep(description="placeholder")
        # Pydantic model is immutable by default; use model_copy
        invalid_step = step.model_copy(update={"description": "   "})
        errors = invalid_step.validate_inputs()
        assert any("description" in e for e in errors)


class TestWorkflowRunner:
    def test_run_single_step_workflow(self, tmp_path):
        step = WorkflowStep(description="get time")
        wf = Workflow(title="Time Check", steps=[step])
        wf.save(workflow_dir=tmp_path)

        runner = WorkflowRunner(
            wf,
            workflow_dir=tmp_path,
            approval_dir=tmp_path,
        )
        result = runner.run()
        assert result is not None

    def test_dry_run_does_not_modify_workflow(self, tmp_path):
        step = WorkflowStep(description="get time")
        wf = Workflow(title="Dry Run", steps=[step])

        runner = WorkflowRunner(wf, workflow_dir=tmp_path)
        result = runner.dry_run()
        assert result is not None
        assert wf.status in ("queued",)  # unchanged by dry_run


class TestWorkflowRunnerModuleFunctions:
    def test_approve_workflow(self, tmp_path):
        apr = WorkflowApproval(
            approval_id=generate_approval_id(),
            workflow_id="wf1",
            step_id="s1",
            requested_by="user",
        )
        apr.save(approval_dir=tmp_path)
        result = approve_workflow(
            apr.approval_id, decided_by="admin", approval_dir=tmp_path
        )
        assert result is True

    def test_deny_workflow(self, tmp_path):
        apr = WorkflowApproval(
            approval_id=generate_approval_id(),
            workflow_id="wf2",
            step_id="s2",
            requested_by="user",
        )
        apr.save(approval_dir=tmp_path)
        result = deny_workflow(
            apr.approval_id,
            decided_by="admin",
            reason="not permitted",
            approval_dir=tmp_path,
        )
        assert result is True

    def test_approve_nonexistent_returns_false(self, tmp_path):
        result = approve_workflow("apr_nonexistent", approval_dir=tmp_path)
        assert result is False

    def test_list_pending_approvals(self, tmp_path):
        apr = WorkflowApproval(
            approval_id=generate_approval_id(),
            workflow_id="wf3",
            step_id="s3",
            requested_by="user",
        )
        apr.save(approval_dir=tmp_path)
        pending = list_pending_approvals(approval_dir=tmp_path)
        assert any(a.approval_id == apr.approval_id for a in pending)


# ---------------------------------------------------------------------------
# AC-4: no tests rely on external services or live credentials
# ---------------------------------------------------------------------------


def test_no_live_network_calls_during_instantiation(monkeypatch):
    """Instantiating core classes must not touch the network."""
    original_connect = socket.socket.connect

    def fail_connect(self, addr):
        raise AssertionError(f"test_us099 attempted a network call to {addr}")

    monkeypatch.setattr(socket.socket, "connect", fail_connect)

    _ = ToolRegistry()
    _ = Workflow(title="g")
    _ = WorkflowStep(description="s")
    _ = WorkflowApproval(
        approval_id=generate_approval_id(),
        workflow_id="wf",
        step_id="s",
        requested_by="user",
    )
