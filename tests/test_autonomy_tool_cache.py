"""Unit tests for US-235: ToolCache and runner cache integration."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from rex.autonomy.models import Plan, PlanStatus, PlanStep
from rex.autonomy.runner import execute_plan
from rex.autonomy.tool_cache import ToolCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(sid: str, tool: str = "noop", args: dict[str, Any] | None = None) -> PlanStep:
    return PlanStep(id=sid, tool=tool, description=f"Step {sid}", args=args or {})


def _plan(steps: list[PlanStep]) -> Plan:
    return Plan(id="p1", goal="Test goal", steps=steps)


# ---------------------------------------------------------------------------
# ToolCache — basic API
# ---------------------------------------------------------------------------


class TestToolCacheBasic:
    def test_get_miss_returns_none(self) -> None:
        cache = ToolCache()
        assert cache.get("search", {"query": "hello"}) is None

    def test_set_then_get_returns_result(self) -> None:
        cache = ToolCache()
        cache.set("search", {"query": "hello"}, "result_A")
        assert cache.get("search", {"query": "hello"}) == "result_A"

    def test_different_tool_is_miss(self) -> None:
        cache = ToolCache()
        cache.set("search", {"q": "x"}, "r")
        assert cache.get("fetch", {"q": "x"}) is None

    def test_different_args_is_miss(self) -> None:
        cache = ToolCache()
        cache.set("search", {"q": "x"}, "r")
        assert cache.get("search", {"q": "y"}) is None

    def test_empty_args_key(self) -> None:
        cache = ToolCache()
        cache.set("noop", {}, "done")
        assert cache.get("noop", {}) == "done"

    def test_len_reflects_entries(self) -> None:
        cache = ToolCache()
        assert len(cache) == 0
        cache.set("t1", {}, "r1")
        assert len(cache) == 1
        cache.set("t2", {}, "r2")
        assert len(cache) == 2

    def test_overwrite_same_key(self) -> None:
        cache = ToolCache()
        cache.set("t", {"a": 1}, "first")
        cache.set("t", {"a": 1}, "second")
        assert cache.get("t", {"a": 1}) == "second"
        assert len(cache) == 1


# ---------------------------------------------------------------------------
# ToolCache — key ordering
# ---------------------------------------------------------------------------


class TestToolCacheKeyOrdering:
    def test_args_order_independent(self) -> None:
        """Cache key must use sorted args so order does not matter."""
        cache = ToolCache()
        cache.set("t", {"b": 2, "a": 1}, "result")
        assert cache.get("t", {"a": 1, "b": 2}) == "result"

    def test_different_values_distinct_keys(self) -> None:
        cache = ToolCache()
        cache.set("t", {"k": "v1"}, "r1")
        cache.set("t", {"k": "v2"}, "r2")
        assert cache.get("t", {"k": "v1"}) == "r1"
        assert cache.get("t", {"k": "v2"}) == "r2"


# ---------------------------------------------------------------------------
# ToolCache — DEBUG log on cache hit
# ---------------------------------------------------------------------------


class TestToolCacheLogging:
    def test_cache_hit_logged_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        cache = ToolCache()
        cache.set("search", {"q": "hi"}, "result")
        with caplog.at_level(logging.DEBUG, logger="rex.autonomy.tool_cache"):
            cache.get("search", {"q": "hi"})
        assert any("Tool cache hit" in r.message for r in caplog.records)

    def test_cache_miss_not_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        cache = ToolCache()
        with caplog.at_level(logging.DEBUG, logger="rex.autonomy.tool_cache"):
            cache.get("search", {"q": "hi"})
        assert not any("Tool cache hit" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Runner cache integration — same call twice is a cache hit
# ---------------------------------------------------------------------------


class TestRunnerCacheIntegration:
    def test_same_tool_args_called_once(self) -> None:
        """Two steps with identical tool+args should only call the tool once."""
        call_log: list[str] = []

        def _tracked(**kw: Any) -> str:
            call_log.append(str(kw))
            return "done"

        plan = _plan(
            [
                _step("s1", tool="noop", args={"x": 1}),
                _step("s2", tool="noop", args={"x": 1}),
            ]
        )
        execute_plan(plan, {"noop": _tracked})

        assert len(call_log) == 1  # second call served from cache

    def test_different_args_both_called(self) -> None:
        """Two steps with different args should each call the tool."""
        call_log: list[str] = []

        def _tracked(**kw: Any) -> str:
            call_log.append(str(kw))
            return "done"

        plan = _plan(
            [
                _step("s1", tool="noop", args={"x": 1}),
                _step("s2", tool="noop", args={"x": 2}),
            ]
        )
        execute_plan(plan, {"noop": _tracked})

        assert len(call_log) == 2

    def test_different_tools_both_called(self) -> None:
        call_log: list[str] = []

        def _a(**_: Any) -> str:
            call_log.append("a")
            return "done"

        def _b(**_: Any) -> str:
            call_log.append("b")
            return "done"

        plan = _plan(
            [
                _step("s1", tool="a", args={}),
                _step("s2", tool="b", args={}),
            ]
        )
        execute_plan(plan, {"a": _a, "b": _b})

        assert call_log == ["a", "b"]

    def test_cache_not_shared_across_runs(self) -> None:
        """A new cache is created for each execute_plan call."""
        call_log: list[int] = []

        def _tracked(**_: Any) -> str:
            call_log.append(1)
            return "done"

        plan1 = _plan([_step("s1", tool="noop", args={"x": 1})])
        plan2 = _plan([_step("s1", tool="noop", args={"x": 1})])

        execute_plan(plan1, {"noop": _tracked})
        execute_plan(plan2, {"noop": _tracked})

        assert len(call_log) == 2  # each run calls the tool independently

    def test_cached_result_propagated_to_step(self) -> None:
        """Second step with same args gets the cached result."""

        def _noop(**_: Any) -> str:
            return "cached_value"

        plan = _plan(
            [
                _step("s1", tool="noop", args={}),
                _step("s2", tool="noop", args={}),
            ]
        )
        execute_plan(plan, {"noop": _noop})

        assert plan.steps[0].result == "cached_value"
        assert plan.steps[1].result == "cached_value"

    def test_plan_completes_with_cache(self) -> None:
        def _noop(**_: Any) -> str:
            return "ok"

        plan = _plan(
            [
                _step("s1", tool="noop", args={"k": "v"}),
                _step("s2", tool="noop", args={"k": "v"}),
            ]
        )
        result = execute_plan(plan, {"noop": _noop})
        assert result.status == PlanStatus.COMPLETED
