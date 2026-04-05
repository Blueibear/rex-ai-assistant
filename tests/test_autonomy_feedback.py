"""Unit tests for rex.autonomy.feedback — FeedbackAnalyzer."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from rex.autonomy.feedback import FeedbackAnalyzer
from rex.autonomy.history import ExecutionRecord, HistoryStore
from rex.autonomy.models import Plan, PlanStep, StepStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_backend(response: str = "Summary text.") -> MagicMock:
    backend = MagicMock()
    backend.generate.return_value = response
    return backend


def _make_step(tool: str, status: StepStatus = StepStatus.SUCCESS) -> PlanStep:
    step = PlanStep(id=str(uuid.uuid4()), tool=tool, description=f"Call {tool}")
    step.status = status
    return step


def _make_record(
    goal: str = "Some goal",
    outcome: str = "success",
    steps: list[PlanStep] | None = None,
    replan_count: int = 0,
    error_summary: str | None = None,
) -> ExecutionRecord:
    if steps is None:
        steps = [_make_step("noop")]
    plan = Plan(id=str(uuid.uuid4()), goal=goal, steps=steps)
    return ExecutionRecord(
        goal=goal,
        plan=plan,
        outcome=outcome,  # type: ignore[arg-type]
        duration_s=1.0,
        replan_count=replan_count,
        error_summary=error_summary,
        timestamp=datetime.now(UTC),
    )


def _store_with_records(records: list[ExecutionRecord]) -> HistoryStore:
    """Return a HistoryStore whose recent() method yields *records*."""
    store = MagicMock(spec=HistoryStore)
    store.recent = AsyncMock(return_value=records)
    return store


# ---------------------------------------------------------------------------
# Empty history
# ---------------------------------------------------------------------------


class TestFeedbackAnalyzerEmptyHistory:
    def test_returns_empty_string_when_no_records(self) -> None:
        store = _store_with_records([])
        analyzer = FeedbackAnalyzer(backend=_mock_backend())

        result = analyzer.summarize("Do something", store)

        assert result == ""

    def test_llm_not_called_when_no_records(self) -> None:
        backend = _mock_backend()
        store = _store_with_records([])
        analyzer = FeedbackAnalyzer(backend=backend)

        analyzer.summarize("Do something", store)

        backend.generate.assert_not_called()

    def test_recent_called_with_n_10(self) -> None:
        store = _store_with_records([])
        analyzer = FeedbackAnalyzer(backend=_mock_backend())

        analyzer.summarize("goal", store)

        store.recent.assert_called_once_with(n=10)


# ---------------------------------------------------------------------------
# Non-empty history
# ---------------------------------------------------------------------------


class TestFeedbackAnalyzerWithHistory:
    def test_returns_non_empty_string(self) -> None:
        records = [_make_record()]
        store = _store_with_records(records)
        analyzer = FeedbackAnalyzer(backend=_mock_backend("Good summary here."))

        result = analyzer.summarize("Do something", store)

        assert result != ""

    def test_returns_llm_response(self) -> None:
        records = [_make_record()]
        store = _store_with_records(records)
        analyzer = FeedbackAnalyzer(backend=_mock_backend("Specific LLM output."))

        result = analyzer.summarize("goal", store)

        assert result == "Specific LLM output."

    def test_llm_called_once(self) -> None:
        records = [_make_record(), _make_record()]
        store = _store_with_records(records)
        backend = _mock_backend()
        analyzer = FeedbackAnalyzer(backend=backend)

        analyzer.summarize("goal", store)

        backend.generate.assert_called_once()

    def test_prompt_contains_current_goal(self) -> None:
        records = [_make_record()]
        store = _store_with_records(records)
        backend = _mock_backend()
        analyzer = FeedbackAnalyzer(backend=backend)

        analyzer.summarize("My unique planning goal XYZ", store)

        call_kwargs = backend.generate.call_args
        prompt = call_kwargs.kwargs["messages"][0]["content"]
        assert "My unique planning goal XYZ" in prompt

    def test_prompt_contains_record_goal(self) -> None:
        records = [_make_record(goal="Earlier accomplished goal")]
        store = _store_with_records(records)
        backend = _mock_backend()
        analyzer = FeedbackAnalyzer(backend=backend)

        analyzer.summarize("new goal", store)

        call_kwargs = backend.generate.call_args
        prompt = call_kwargs.kwargs["messages"][0]["content"]
        assert "Earlier accomplished goal" in prompt

    def test_prompt_contains_outcome(self) -> None:
        records = [_make_record(outcome="failed", error_summary="Tool X crashed")]
        store = _store_with_records(records)
        backend = _mock_backend()
        analyzer = FeedbackAnalyzer(backend=backend)

        analyzer.summarize("goal", store)

        call_kwargs = backend.generate.call_args
        prompt = call_kwargs.kwargs["messages"][0]["content"]
        assert "failed" in prompt

    def test_prompt_contains_error_summary(self) -> None:
        records = [_make_record(outcome="failed", error_summary="Timeout on step 2")]
        store = _store_with_records(records)
        backend = _mock_backend()
        analyzer = FeedbackAnalyzer(backend=backend)

        analyzer.summarize("goal", store)

        call_kwargs = backend.generate.call_args
        prompt = call_kwargs.kwargs["messages"][0]["content"]
        assert "Timeout on step 2" in prompt

    def test_prompt_contains_tool_names(self) -> None:
        records = [_make_record(steps=[_make_step("search_web"), _make_step("send_email")])]
        store = _store_with_records(records)
        backend = _mock_backend()
        analyzer = FeedbackAnalyzer(backend=backend)

        analyzer.summarize("goal", store)

        call_kwargs = backend.generate.call_args
        prompt = call_kwargs.kwargs["messages"][0]["content"]
        assert "search_web" in prompt
        assert "send_email" in prompt

    def test_multiple_records_all_included_in_prompt(self) -> None:
        records = [
            _make_record(goal="goal-A"),
            _make_record(goal="goal-B"),
            _make_record(goal="goal-C"),
        ]
        store = _store_with_records(records)
        backend = _mock_backend()
        analyzer = FeedbackAnalyzer(backend=backend)

        analyzer.summarize("current goal", store)

        call_kwargs = backend.generate.call_args
        prompt = call_kwargs.kwargs["messages"][0]["content"]
        assert "goal-A" in prompt
        assert "goal-B" in prompt
        assert "goal-C" in prompt

    def test_failed_tools_highlighted_in_prompt(self) -> None:
        steps = [
            _make_step("ok_tool", StepStatus.SUCCESS),
            _make_step("bad_tool", StepStatus.FAILED),
        ]
        records = [_make_record(steps=steps, outcome="partial")]
        store = _store_with_records(records)
        backend = _mock_backend()
        analyzer = FeedbackAnalyzer(backend=backend)

        analyzer.summarize("goal", store)

        call_kwargs = backend.generate.call_args
        prompt = call_kwargs.kwargs["messages"][0]["content"]
        assert "bad_tool" in prompt
