"""Tests for US-325: Autonomous workflows -- minimal scheduled task runner."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path


def _make_config(tmp_path: Path, tasks: list[dict]) -> Path:
    config_path = tmp_path / "rex_config.json"
    config_path.write_text(json.dumps({"workflows": {"tasks": tasks}}), encoding="utf-8")
    return config_path


def _make_runner(tmp_path: Path, tasks: list[dict]):
    from rex.workflow_runner import ScheduledTaskRunner

    config_path = _make_config(tmp_path, tasks)
    state_path = tmp_path / "state.json"
    return ScheduledTaskRunner(config_path=config_path, state_path=state_path)


def test_load_tasks_from_config(tmp_path: Path) -> None:
    """Tasks defined in config are loaded correctly."""
    runner = _make_runner(
        tmp_path,
        [{"name": "t1", "schedule": "interval:3600", "action": "doctor", "enabled": True}],
    )
    tasks = runner.tasks()
    assert len(tasks) == 1
    assert tasks[0].name == "t1"
    assert tasks[0].schedule == "interval:3600"
    assert tasks[0].action == "doctor"
    assert tasks[0].enabled is True


def test_example_task_included_when_no_config(tmp_path: Path) -> None:
    """Falls back to built-in example tasks when config has no tasks."""
    from rex.workflow_runner import ScheduledTaskRunner

    config_path = tmp_path / "empty.json"
    config_path.write_text("{}", encoding="utf-8")
    runner = ScheduledTaskRunner(config_path=config_path, state_path=tmp_path / "state.json")
    names = [t.name for t in runner.tasks()]
    assert "daily_weather" in names


def test_disabled_task_not_run(tmp_path: Path) -> None:
    """Disabled tasks are not executed even if due."""
    runner = _make_runner(
        tmp_path,
        [{"name": "disabled", "schedule": "interval:1", "action": "doctor", "enabled": False}],
    )
    executed = runner.run_due_tasks()
    assert "disabled" not in executed


def test_due_task_is_triggered(tmp_path: Path) -> None:
    """A task with no previous run is due and gets triggered."""
    runner = _make_runner(
        tmp_path,
        [{"name": "t1", "schedule": "interval:3600", "action": "doctor", "enabled": True}],
    )
    # Patch _run_task to avoid subprocess
    ran: list[str] = []
    runner._run_task = lambda task: ran.append(task.name) or True  # type: ignore[method-assign]
    executed = runner.run_due_tasks()
    assert "t1" in executed
    assert "t1" in ran


def test_recent_task_not_due(tmp_path: Path) -> None:
    """A task run recently is not triggered again before its interval."""
    runner = _make_runner(
        tmp_path,
        [{"name": "t1", "schedule": "interval:3600", "action": "doctor", "enabled": True}],
    )
    runner._run_task = lambda task: True  # type: ignore[method-assign]
    # Seed state as if it just ran
    runner._state["t1"] = datetime.now(UTC).isoformat()
    runner._save_state()
    executed = runner.run_due_tasks()
    assert "t1" not in executed


def test_overdue_task_is_triggered(tmp_path: Path) -> None:
    """A task whose last run was longer ago than its interval is triggered."""
    runner = _make_runner(
        tmp_path,
        [{"name": "t1", "schedule": "interval:3600", "action": "doctor", "enabled": True}],
    )
    ran: list[str] = []
    runner._run_task = lambda task: ran.append(task.name) or True  # type: ignore[method-assign]
    runner._state["t1"] = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    runner._save_state()
    executed = runner.run_due_tasks()
    assert "t1" in executed
    assert "t1" in ran


def test_failed_task_does_not_block_others(tmp_path: Path) -> None:
    """A task that fails does not prevent subsequent tasks from running."""
    runner = _make_runner(
        tmp_path,
        [
            {"name": "fail", "schedule": "interval:1", "action": "doctor", "enabled": True},
            {"name": "pass", "schedule": "interval:1", "action": "doctor", "enabled": True},
        ],
    )
    ran: list[str] = []

    def fake_run(task):
        ran.append(task.name)
        if task.name == "fail":
            raise RuntimeError("boom")
        return True

    runner._run_task = fake_run  # type: ignore[method-assign]
    executed = runner.run_due_tasks()
    assert "fail" in executed
    assert "pass" in executed
    assert "pass" in ran


def test_state_persisted_after_run(tmp_path: Path) -> None:
    """Last-run timestamp is written to disk after a task executes."""
    runner = _make_runner(
        tmp_path,
        [{"name": "t1", "schedule": "interval:1", "action": "doctor", "enabled": True}],
    )
    runner._run_task = lambda task: True  # type: ignore[method-assign]
    runner.run_due_tasks()
    state = json.loads(runner.state_path.read_text(encoding="utf-8"))
    assert "t1" in state


def test_example_config_has_daily_weather() -> None:
    """rex_config.example.json must contain the daily_weather example task."""
    config = Path(__file__).parent.parent / "config" / "rex_config.example.json"
    data = json.loads(config.read_text(encoding="utf-8"))
    tasks = data.get("workflows", {}).get("tasks", [])
    names = [t.get("name") for t in tasks]
    assert "daily_weather" in names
