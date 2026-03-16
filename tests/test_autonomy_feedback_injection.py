"""Unit tests for US-228: feedback_summary injection into LLMPlanner and runner."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from rex.autonomy.llm_planner import LLMPlanner, ToolDefinition
from rex.autonomy.runner import run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_backend(steps: list[str] | None = None) -> MagicMock:
    """Backend whose generate() returns a valid single-step plan JSON."""
    tool = (steps or ["noop"])[0]
    response = json.dumps([{"tool": tool, "args": {}, "description": f"Call {tool}"}])
    backend = MagicMock()
    backend.generate.return_value = response
    return backend


def _captured_prompt(backend: MagicMock) -> str:
    """Extract the prompt string from the first backend.generate() call."""
    call_kwargs = backend.generate.call_args
    return call_kwargs.kwargs["messages"][0]["content"]


# ---------------------------------------------------------------------------
# LLMPlanner.plan() — feedback_summary in prompt
# ---------------------------------------------------------------------------


class TestLLMPlannerFeedbackSummary:
    def test_prompt_contains_feedback_section_when_provided(self) -> None:
        backend = _mock_backend()
        planner = LLMPlanner(tools=[], backend=backend)

        planner.plan("My goal", {}, feedback_summary="Tools X and Y failed before.")

        prompt = _captured_prompt(backend)
        assert "## Past Execution Patterns" in prompt

    def test_prompt_contains_summary_text_when_provided(self) -> None:
        backend = _mock_backend()
        planner = LLMPlanner(tools=[], backend=backend)

        planner.plan("My goal", {}, feedback_summary="Tools X and Y failed before.")

        prompt = _captured_prompt(backend)
        assert "Tools X and Y failed before." in prompt

    def test_prompt_omits_feedback_section_when_empty(self) -> None:
        backend = _mock_backend()
        planner = LLMPlanner(tools=[], backend=backend)

        planner.plan("My goal", {}, feedback_summary="")

        prompt = _captured_prompt(backend)
        assert "## Past Execution Patterns" not in prompt

    def test_prompt_omits_feedback_section_by_default(self) -> None:
        backend = _mock_backend()
        planner = LLMPlanner(tools=[], backend=backend)

        planner.plan("My goal", {})

        prompt = _captured_prompt(backend)
        assert "## Past Execution Patterns" not in prompt

    def test_plan_still_returned_with_feedback(self) -> None:
        from rex.autonomy.models import Plan

        backend = _mock_backend()
        planner = LLMPlanner(tools=[], backend=backend)

        result = planner.plan("My goal", {}, feedback_summary="Some summary.")

        assert isinstance(result, Plan)

    def test_feedback_section_appears_before_goal_section(self) -> None:
        backend = _mock_backend()
        planner = LLMPlanner(tools=[], backend=backend)

        planner.plan("Goal text here", {}, feedback_summary="Summary here.")

        prompt = _captured_prompt(backend)
        patterns_idx = prompt.index("## Past Execution Patterns")
        goal_idx = prompt.index("## Goal")
        assert patterns_idx < goal_idx


# ---------------------------------------------------------------------------
# runner.run() — calls FeedbackAnalyzer before LLMPlanner
# ---------------------------------------------------------------------------


class TestRunnerFeedbackInjection:
    def test_runner_calls_summarize_when_analyzer_and_store_provided(self) -> None:
        backend = _mock_backend()
        analyzer = MagicMock()
        analyzer.summarize.return_value = ""
        store = MagicMock()

        with patch("rex.autonomy.runner.LLMPlanner") as mock_cls:
            from rex.autonomy.llm_planner import LLMPlanner as RealLLM

            instance = MagicMock(spec=RealLLM)
            instance.plan.return_value = MagicMock()
            mock_cls.return_value = instance

            run(
                "Do something",
                feedback_analyzer=analyzer,
                history_store=store,
                planner_key="llm",
            )

        analyzer.summarize.assert_called_once()

    def test_runner_does_not_call_summarize_without_store(self) -> None:
        analyzer = MagicMock()
        analyzer.summarize.return_value = "feedback"

        with patch("rex.autonomy.runner.LLMPlanner") as mock_cls:
            from rex.autonomy.llm_planner import LLMPlanner as RealLLM

            instance = MagicMock(spec=RealLLM)
            instance.plan.return_value = MagicMock()
            mock_cls.return_value = instance

            run("Do something", feedback_analyzer=analyzer, planner_key="llm")

        analyzer.summarize.assert_not_called()

    def test_runner_does_not_call_summarize_without_analyzer(self) -> None:
        store = MagicMock()

        with patch("rex.autonomy.runner.LLMPlanner") as mock_cls:
            from rex.autonomy.llm_planner import LLMPlanner as RealLLM

            instance = MagicMock(spec=RealLLM)
            instance.plan.return_value = MagicMock()
            mock_cls.return_value = instance

            run("Do something", history_store=store, planner_key="llm")

        # No summarize call expected — just verify no crash
        assert True

    def test_runner_passes_feedback_to_planner(self) -> None:
        analyzer = MagicMock()
        analyzer.summarize.return_value = "Previous run used tool_a."
        store = MagicMock()

        with patch("rex.autonomy.runner.LLMPlanner") as mock_cls:
            from rex.autonomy.llm_planner import LLMPlanner as RealLLM

            instance = MagicMock(spec=RealLLM)
            instance.plan.return_value = MagicMock()
            mock_cls.return_value = instance

            run(
                "Do something",
                feedback_analyzer=analyzer,
                history_store=store,
                planner_key="llm",
            )

            call_kwargs = instance.plan.call_args
            assert call_kwargs.kwargs.get("feedback_summary") == "Previous run used tool_a."

    def test_runner_summarize_failure_does_not_raise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        analyzer = MagicMock()
        analyzer.summarize.side_effect = RuntimeError("LLM unavailable")
        store = MagicMock()

        with patch("rex.autonomy.runner.LLMPlanner") as mock_cls:
            from rex.autonomy.llm_planner import LLMPlanner as RealLLM

            instance = MagicMock(spec=RealLLM)
            instance.plan.return_value = MagicMock()
            mock_cls.return_value = instance

            with caplog.at_level(logging.WARNING, logger="rex.autonomy.runner"):
                result = run(
                    "Do something",
                    feedback_analyzer=analyzer,
                    history_store=store,
                    planner_key="llm",
                )

        assert result is not None

    def test_runner_summarize_failure_logged_as_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        analyzer = MagicMock()
        analyzer.summarize.side_effect = RuntimeError("LLM down")
        store = MagicMock()

        with patch("rex.autonomy.runner.LLMPlanner") as mock_cls:
            from rex.autonomy.llm_planner import LLMPlanner as RealLLM

            instance = MagicMock(spec=RealLLM)
            instance.plan.return_value = MagicMock()
            mock_cls.return_value = instance

            with caplog.at_level(logging.WARNING, logger="rex.autonomy.runner"):
                run(
                    "Do something",
                    feedback_analyzer=analyzer,
                    history_store=store,
                    planner_key="llm",
                )

        assert any("feedback summarize failed" in r.message for r in caplog.records)
