"""Unit tests for PreferenceLearner (US-238)."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rex.autonomy.history import ExecutionRecord
from rex.autonomy.models import Plan, PlanStatus, PlanStep, StepStatus
from rex.autonomy.preference_learner import PreferenceLearner
from rex.autonomy.preferences import PreferenceStore, UserPreferenceProfile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plan(goal: str = "test goal", cost: float = 0.0) -> Plan:
    step = PlanStep(id="s1", tool="noop", description="do noop", args={})
    step.status = StepStatus.SUCCESS
    plan = Plan(id="p1", goal=goal, steps=[step])
    plan.status = PlanStatus.COMPLETED
    return plan


def _make_record(goal: str = "test goal", cost: float = 0.05) -> ExecutionRecord:
    plan = _make_plan(goal=goal, cost=cost)
    return ExecutionRecord(
        goal=goal,
        plan=plan,
        outcome="success",
        duration_s=1.0,
        total_cost_usd=cost,
    )


def _default_profile() -> UserPreferenceProfile:
    return UserPreferenceProfile()


# ---------------------------------------------------------------------------
# active_hours tests
# ---------------------------------------------------------------------------


def test_active_hours_appended_when_not_present() -> None:
    learner = PreferenceLearner()
    profile = _default_profile()
    record = _make_record()

    fixed_hour = 14
    fixed_dt = datetime(2024, 1, 1, fixed_hour, 0, 0, tzinfo=timezone.utc)
    with patch("rex.autonomy.preference_learner.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_dt
        updated = learner.update(record, profile)

    assert fixed_hour in updated.active_hours


def test_active_hours_not_duplicated() -> None:
    learner = PreferenceLearner()
    profile = _default_profile()
    profile.active_hours = [14]
    record = _make_record()

    fixed_dt = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
    with patch("rex.autonomy.preference_learner.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_dt
        updated = learner.update(record, profile)

    assert updated.active_hours.count(14) == 1


def test_active_hours_multiple_different_hours() -> None:
    learner = PreferenceLearner()
    profile = _default_profile()
    record = _make_record()

    for hour in (9, 17):
        fixed_dt = datetime(2024, 1, 1, hour, 0, 0, tzinfo=timezone.utc)
        with patch("rex.autonomy.preference_learner.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_dt
            profile = learner.update(record, profile)

    assert 9 in profile.active_hours
    assert 17 in profile.active_hours


# ---------------------------------------------------------------------------
# avg_budget_usd tests
# ---------------------------------------------------------------------------


def test_avg_budget_initialised_from_first_record() -> None:
    learner = PreferenceLearner()
    profile = _default_profile()
    record = _make_record(cost=0.10)

    with patch("rex.autonomy.preference_learner.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        updated = learner.update(record, profile)

    assert updated.avg_budget_usd == pytest.approx(0.10)


def test_avg_budget_ema_applied_on_subsequent_records() -> None:
    learner = PreferenceLearner()
    profile = _default_profile()
    profile.avg_budget_usd = 0.10

    record = _make_record(cost=0.20)

    fixed_dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    with patch("rex.autonomy.preference_learner.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_dt
        updated = learner.update(record, profile)

    # α=0.2: 0.2*0.20 + 0.8*0.10 = 0.04 + 0.08 = 0.12
    assert updated.avg_budget_usd == pytest.approx(0.12)


def test_avg_budget_zero_cost_record() -> None:
    learner = PreferenceLearner()
    profile = _default_profile()
    record = _make_record(cost=0.0)

    with patch("rex.autonomy.preference_learner.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        updated = learner.update(record, profile)

    assert updated.avg_budget_usd == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# common_goal_patterns tests
# ---------------------------------------------------------------------------


def test_goal_pattern_added_to_empty_list() -> None:
    learner = PreferenceLearner()
    profile = _default_profile()
    record = _make_record(goal="search the web")

    with patch("rex.autonomy.preference_learner.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        updated = learner.update(record, profile)

    assert "search the web" in updated.common_goal_patterns


def test_goal_pattern_deduplicated_and_moved_to_front() -> None:
    learner = PreferenceLearner()
    profile = _default_profile()
    profile.common_goal_patterns = ["old goal", "search the web", "another goal"]
    record = _make_record(goal="search the web")

    with patch("rex.autonomy.preference_learner.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        updated = learner.update(record, profile)

    assert updated.common_goal_patterns[0] == "search the web"
    assert updated.common_goal_patterns.count("search the web") == 1


def test_goal_patterns_capped_at_20_entries() -> None:
    learner = PreferenceLearner()
    profile = _default_profile()
    profile.common_goal_patterns = [f"goal {i}" for i in range(20)]
    record = _make_record(goal="new goal")

    with patch("rex.autonomy.preference_learner.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        updated = learner.update(record, profile)

    assert len(updated.common_goal_patterns) == 20
    assert updated.common_goal_patterns[0] == "new goal"
    # Oldest entry (goal 19) should be evicted.
    assert "goal 19" not in updated.common_goal_patterns


def test_empty_goal_not_added_to_patterns() -> None:
    learner = PreferenceLearner()
    profile = _default_profile()
    record = _make_record(goal="   ")  # whitespace-only goal

    with patch("rex.autonomy.preference_learner.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        updated = learner.update(record, profile)

    assert updated.common_goal_patterns == []


# ---------------------------------------------------------------------------
# Immutability — original profile must not be mutated
# ---------------------------------------------------------------------------


def test_update_returns_new_profile_does_not_mutate_original() -> None:
    learner = PreferenceLearner()
    profile = _default_profile()
    profile.common_goal_patterns = ["existing goal"]
    original_patterns = list(profile.common_goal_patterns)

    record = _make_record(goal="new goal")

    with patch("rex.autonomy.preference_learner.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        updated = learner.update(record, profile)

    assert profile.common_goal_patterns == original_patterns
    assert updated is not profile


# ---------------------------------------------------------------------------
# last_updated is refreshed
# ---------------------------------------------------------------------------


def test_last_updated_set_to_now() -> None:
    learner = PreferenceLearner()
    profile = _default_profile()
    record = _make_record()

    fixed_dt = datetime(2030, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    with patch("rex.autonomy.preference_learner.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_dt
        updated = learner.update(record, profile)

    assert updated.last_updated == fixed_dt


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


def test_runner_calls_learner_after_successful_plan() -> None:
    """execute_plan calls PreferenceLearner.update + PreferenceStore.save on success."""
    from rex.autonomy.runner import execute_plan

    plan = _make_plan()

    learner = MagicMock(spec=PreferenceLearner)
    fake_profile = _default_profile()
    learner.update.return_value = fake_profile

    with tempfile.TemporaryDirectory() as tmp:
        store = PreferenceStore(prefs_path=Path(tmp) / "prefs.json")

        result = execute_plan(
            plan,
            {"noop": lambda **kw: "ok"},
            preference_learner=learner,
            preference_store=store,
        )

        assert result.status == PlanStatus.COMPLETED
        learner.update.assert_called_once()
        # Profile should have been saved.
        assert (Path(tmp) / "prefs.json").exists()


def test_runner_does_not_call_learner_on_failed_plan() -> None:
    """execute_plan must NOT call PreferenceLearner.update when the plan fails."""
    from rex.autonomy.runner import execute_plan

    step = PlanStep(id="s1", tool="missing_tool", description="will fail", args={})
    plan = Plan(id="p1", goal="fail goal", steps=[step])

    learner = MagicMock(spec=PreferenceLearner)

    result = execute_plan(
        plan,
        {},  # no tools — step will fail
        preference_learner=learner,
        preference_store=MagicMock(spec=PreferenceStore),
    )

    assert result.status == PlanStatus.FAILED
    learner.update.assert_not_called()


def test_runner_works_without_learner() -> None:
    """execute_plan must succeed when no preference_learner is supplied."""
    from rex.autonomy.runner import execute_plan

    plan = _make_plan()
    result = execute_plan(plan, {"noop": lambda **kw: "ok"})
    assert result.status == PlanStatus.COMPLETED
