"""US-005: Smoke tests for rex_tasks_bridge, rex_reminders_bridge, rex_memories_bridge.

Verifies:
- Each bridge accepts JSON on stdin with {"action": "list"} and returns valid JSON
- Each bridge returns {"error": "..."} on invalid (non-JSON) input
"""
from __future__ import annotations

import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest


def _run_main(module_name: str, stdin_data: str) -> dict:
    """Import a bridge module and invoke main() with the given stdin, returning parsed output."""
    import importlib

    mod = importlib.import_module(module_name.replace(".py", "").replace("/", "."))

    captured = StringIO()
    with patch("sys.stdin", StringIO(stdin_data)):
        with patch("sys.stdout", captured):
            try:
                mod.main()
            except SystemExit:
                pass

    output = captured.getvalue().strip()
    assert output, f"{module_name}: no output produced"
    return json.loads(output)


# ---------------------------------------------------------------------------
# rex_tasks_bridge
# ---------------------------------------------------------------------------

TASKS_MODULE = "rex_tasks_bridge"


class TestTasksBridgeJsonIO:
    def test_action_list_returns_valid_json(self):
        mock_scheduler = MagicMock()
        mock_scheduler.list_jobs.return_value = []
        with patch.dict("sys.modules", {"rex.scheduler": MagicMock(get_scheduler=lambda: mock_scheduler)}):
            result = _run_main(TASKS_MODULE, '{"action": "list"}')
        assert isinstance(result, dict)
        assert "tasks" in result or "error" in result

    def test_command_list_returns_valid_json(self):
        mock_scheduler = MagicMock()
        mock_scheduler.list_jobs.return_value = []
        with patch.dict("sys.modules", {"rex.scheduler": MagicMock(get_scheduler=lambda: mock_scheduler)}):
            result = _run_main(TASKS_MODULE, '{"command": "list"}')
        assert isinstance(result, dict)
        assert "tasks" in result or "error" in result

    def test_invalid_input_returns_error_key(self):
        result = _run_main(TASKS_MODULE, "notjson")
        assert "error" in result

    def test_unknown_action_returns_error_key(self):
        result = _run_main(TASKS_MODULE, '{"action": "bogus"}')
        assert "error" in result


# ---------------------------------------------------------------------------
# rex_reminders_bridge
# ---------------------------------------------------------------------------

REMINDERS_MODULE = "rex_reminders_bridge"


class TestRemindersBridgeJsonIO:
    def test_action_list_returns_valid_json(self):
        mock_service = MagicMock()
        mock_service.list_reminders.return_value = []
        with patch.dict(
            "sys.modules",
            {"rex.reminder_service": MagicMock(get_reminder_service=lambda: mock_service)},
        ):
            result = _run_main(REMINDERS_MODULE, '{"action": "list"}')
        assert isinstance(result, dict)
        assert "reminders" in result or "error" in result

    def test_command_list_returns_valid_json(self):
        mock_service = MagicMock()
        mock_service.list_reminders.return_value = []
        with patch.dict(
            "sys.modules",
            {"rex.reminder_service": MagicMock(get_reminder_service=lambda: mock_service)},
        ):
            result = _run_main(REMINDERS_MODULE, '{"command": "list"}')
        assert isinstance(result, dict)
        assert "reminders" in result or "error" in result

    def test_invalid_input_returns_error_key(self):
        result = _run_main(REMINDERS_MODULE, "notjson")
        assert "error" in result

    def test_unknown_action_returns_error_key(self):
        result = _run_main(REMINDERS_MODULE, '{"action": "bogus"}')
        assert "error" in result


# ---------------------------------------------------------------------------
# rex_memories_bridge
# ---------------------------------------------------------------------------

MEMORIES_MODULE = "rex_memories_bridge"


class TestMemoriesBridgeJsonIO:
    def test_action_list_returns_valid_json(self):
        mock_ltm = MagicMock()
        mock_ltm.search.return_value = []
        with patch.dict(
            "sys.modules",
            {"rex.memory": MagicMock(get_long_term_memory=lambda: mock_ltm)},
        ):
            result = _run_main(MEMORIES_MODULE, '{"action": "list"}')
        assert isinstance(result, dict)
        assert "memories" in result or "error" in result

    def test_command_list_returns_valid_json(self):
        mock_ltm = MagicMock()
        mock_ltm.search.return_value = []
        with patch.dict(
            "sys.modules",
            {"rex.memory": MagicMock(get_long_term_memory=lambda: mock_ltm)},
        ):
            result = _run_main(MEMORIES_MODULE, '{"command": "list"}')
        assert isinstance(result, dict)
        assert "memories" in result or "error" in result

    def test_invalid_input_returns_error_key(self):
        result = _run_main(MEMORIES_MODULE, "notjson")
        assert "error" in result

    def test_unknown_action_returns_error_key(self):
        result = _run_main(MEMORIES_MODULE, '{"action": "bogus"}')
        assert "error" in result
