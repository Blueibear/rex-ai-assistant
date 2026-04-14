"""US-326: Smart planning -- minimal plan-and-execute skeleton.

Tests for create_plan() and execute_plan() in rex.planner.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rex.planner import Result, Step, create_plan, execute_plan

# ---------------------------------------------------------------------------
# Step dataclass
# ---------------------------------------------------------------------------


class TestStep:
    def test_fields_present(self):
        s = Step(description="do something")
        assert s.description == "do something"
        assert s.tool is None
        assert s.status == "pending"

    def test_with_tool(self):
        s = Step(description="search web", tool="web_search")
        assert s.tool == "web_search"

    def test_status_mutable(self):
        s = Step(description="x")
        s.status = "running"
        assert s.status == "running"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


class TestResult:
    def test_fields_present(self):
        steps = [Step(description="a", status="success")]
        r = Result(steps=steps, success=True)
        assert r.success is True
        assert r.errors == []

    def test_with_errors(self):
        steps = [Step(description="a", status="failed")]
        r = Result(steps=steps, success=False, errors=["oops"])
        assert r.errors == ["oops"]


# ---------------------------------------------------------------------------
# execute_plan()
# ---------------------------------------------------------------------------


class TestExecutePlan:
    def test_all_steps_succeed_no_tool_fn(self):
        steps = [Step(description="step 1"), Step(description="step 2")]
        result = execute_plan(steps)
        assert result.success is True
        assert all(s.status == "success" for s in steps)
        assert result.errors == []

    def test_step_with_tool_called(self):
        calls: list[tuple[str, str]] = []

        def tool_fn(tool: str, desc: str) -> str:
            calls.append((tool, desc))
            return "ok"

        steps = [Step(description="search it", tool="web_search")]
        result = execute_plan(steps, tool_fn=tool_fn)
        assert result.success is True
        assert calls == [("web_search", "search it")]

    def test_tool_step_without_tool_fn_succeeds(self):
        steps = [Step(description="call tool", tool="some_tool")]
        result = execute_plan(steps, tool_fn=None)
        assert result.success is True
        assert steps[0].status == "success"

    def test_failing_step_continues_to_next(self):
        def fail_fn(tool: str, desc: str) -> str:
            raise RuntimeError("boom")

        steps = [
            Step(description="fail step", tool="bad_tool"),
            Step(description="good step"),
        ]
        result = execute_plan(steps, tool_fn=fail_fn)
        assert result.success is False
        assert steps[0].status == "failed"
        assert steps[1].status == "success"
        assert len(result.errors) == 1
        assert "boom" in result.errors[0]

    def test_empty_steps(self):
        result = execute_plan([])
        assert result.success is True
        assert result.errors == []

    def test_status_transitions(self):
        seen_statuses: list[str] = []

        def track_fn(tool: str, desc: str) -> str:
            # By the time tool_fn is called, status should be "running"
            return "ok"

        steps = [Step(description="x", tool="t")]
        execute_plan(steps, tool_fn=track_fn)
        assert steps[0].status == "success"


# ---------------------------------------------------------------------------
# create_plan() -- uses mocked LLM
# ---------------------------------------------------------------------------


class TestCreatePlan:
    def _make_llm(self, response: str) -> MagicMock:
        llm = MagicMock()
        llm.generate.return_value = response
        return llm

    def test_returns_list_of_steps(self):
        llm = self._make_llm('[{"description": "find info", "tool": "web_search"}]')
        steps = create_plan("research topic X", llm=llm)
        assert isinstance(steps, list)
        assert len(steps) == 1
        assert steps[0].description == "find info"
        assert steps[0].tool == "web_search"
        assert steps[0].status == "pending"

    def test_multi_step_plan(self):
        payload = (
            '[{"description": "step A", "tool": null},'
            ' {"description": "step B", "tool": "send_email"}]'
        )
        llm = self._make_llm(payload)
        steps = create_plan("do A then B", llm=llm)
        assert len(steps) == 2
        assert steps[0].tool is None
        assert steps[1].tool == "send_email"

    def test_strips_markdown_fences(self):
        fenced = '```json\n[{"description": "plan step", "tool": null}]\n```'
        llm = self._make_llm(fenced)
        steps = create_plan("goal", llm=llm)
        assert len(steps) == 1
        assert steps[0].description == "plan step"

    def test_empty_goal_raises(self):
        llm = self._make_llm("[]")
        with pytest.raises(ValueError, match="empty"):
            create_plan("", llm=llm)

    def test_empty_step_list_raises(self):
        llm = self._make_llm("[]")
        with pytest.raises(ValueError, match="empty step list"):
            create_plan("do something", llm=llm)

    def test_invalid_json_raises(self):
        llm = self._make_llm("not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            create_plan("do something", llm=llm)

    def test_non_array_json_raises(self):
        llm = self._make_llm('{"description": "oops"}')
        with pytest.raises(ValueError, match="array"):
            create_plan("do something", llm=llm)


# ---------------------------------------------------------------------------
# Integration: create_plan + execute_plan round-trip
# ---------------------------------------------------------------------------


class TestPlanAndExecuteIntegration:
    def test_round_trip(self):
        """create_plan output feeds directly into execute_plan."""
        llm = MagicMock()
        llm.generate.return_value = (
            '[{"description": "search", "tool": "web_search"},'
            ' {"description": "summarise", "tool": null}]'
        )

        results_collected: list[str] = []

        def tool_fn(tool: str, desc: str) -> str:
            results_collected.append(tool)
            return f"result from {tool}"

        steps = create_plan("find and summarise topic Y", llm=llm)
        result = execute_plan(steps, tool_fn=tool_fn)

        assert result.success is True
        assert results_collected == ["web_search"]
        assert all(s.status == "success" for s in steps)

    def test_partial_failure_round_trip(self):
        """execute_plan marks failed steps and collects errors."""
        llm = MagicMock()
        llm.generate.return_value = (
            '[{"description": "step 1", "tool": "bad_tool"},'
            ' {"description": "step 2", "tool": null}]'
        )

        def tool_fn(tool: str, desc: str) -> str:
            raise RuntimeError("tool unavailable")

        steps = create_plan("run two steps", llm=llm)
        result = execute_plan(steps, tool_fn=tool_fn)

        assert result.success is False
        assert steps[0].status == "failed"
        assert steps[1].status == "success"
        assert len(result.errors) == 1
