"""US-015: Bridge scripts return traceback-enriched error responses on failure.

AC:
- All bridge scripts wrap execution in try/except and return
  {"error": "<message>", "traceback": "<tb>"} on failure
- GUI backend captures stderr from subprocess calls and includes it in error responses
- CLI mode prints the actual error message, not just "exit code 2"
- Test confirms a deliberately broken bridge returns a readable error
- Typecheck passes
"""

from __future__ import annotations

import importlib
import json
import sys
from io import StringIO
from typing import Any
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_main(mod_name: str, stdin_data: str) -> dict[str, Any]:
    """Import a bridge module, call main() with mocked stdin, return parsed JSON."""
    mod = importlib.import_module(mod_name)
    captured = StringIO()
    with (
        patch.object(sys, "stdin", StringIO(stdin_data)),
        patch.object(sys, "stdout", captured),
        patch("sys.exit"),  # prevent test process from exiting
    ):
        mod.main()
    output = captured.getvalue().strip()
    return json.loads(output)


# ---------------------------------------------------------------------------
# bridge_utils.bridge_error_response
# ---------------------------------------------------------------------------


class TestBridgeErrorResponse:
    def test_returns_ok_false(self) -> None:
        from rex.bridge_utils import bridge_error_response

        result = bridge_error_response(ValueError("boom"))
        assert result["ok"] is False

    def test_error_field_contains_message(self) -> None:
        from rex.bridge_utils import bridge_error_response

        result = bridge_error_response(ValueError("something went wrong"))
        assert "something went wrong" in result["error"]

    def test_traceback_field_present(self) -> None:
        from rex.bridge_utils import bridge_error_response

        try:
            raise RuntimeError("deliberate failure")
        except RuntimeError as exc:
            result = bridge_error_response(exc)

        assert "traceback" in result
        assert result["traceback"]  # non-empty string

    def test_traceback_contains_exception_type(self) -> None:
        from rex.bridge_utils import bridge_error_response

        try:
            raise RuntimeError("deliberate failure")
        except RuntimeError as exc:
            result = bridge_error_response(exc)

        assert "RuntimeError" in result["traceback"]


# ---------------------------------------------------------------------------
# rex_tasks_bridge — deliberately broken inner handler
# ---------------------------------------------------------------------------


class TestTasksBridgeTraceback:
    def test_broken_handler_returns_traceback(self) -> None:
        """When the scheduler raises, the bridge returns error + traceback."""
        import rex_tasks_bridge

        def _broken_scheduler() -> None:
            raise RuntimeError("scheduler exploded")

        with patch.object(
            rex_tasks_bridge, "_handle_list", side_effect=RuntimeError("scheduler exploded")
        ):
            result = _run_main("rex_tasks_bridge", '{"command": "list"}')

        assert result["ok"] is False
        assert "scheduler exploded" in result["error"]
        assert "traceback" in result
        assert result["traceback"]

    def test_traceback_has_exception_class(self) -> None:
        import rex_tasks_bridge

        with patch.object(rex_tasks_bridge, "_handle_list", side_effect=ValueError("bad value")):
            result = _run_main("rex_tasks_bridge", '{"command": "list"}')

        assert "ValueError" in result["traceback"]


# ---------------------------------------------------------------------------
# rex_voices_bridge — deliberately broken inner handler
# ---------------------------------------------------------------------------


class TestVoicesBridgeTraceback:
    def test_broken_list_voices_returns_traceback(self) -> None:
        """When list_voices raises, voices bridge returns traceback."""

        with patch("rex.tts_voices.list_voices", side_effect=ImportError("tts not installed")):
            result = _run_main("rex_voices_bridge", '{"provider": "xtts"}')

        assert result["ok"] is False
        assert "traceback" in result
        assert result["traceback"]


# ---------------------------------------------------------------------------
# rex_memories_bridge — deliberately broken inner handler
# ---------------------------------------------------------------------------


class TestMemoriesBridgeTraceback:
    def test_broken_memory_returns_traceback(self) -> None:
        import rex_memories_bridge

        with patch.object(rex_memories_bridge, "_handle_list", side_effect=OSError("disk full")):
            result = _run_main("rex_memories_bridge", '{"command": "list"}')

        assert result["ok"] is False
        assert "disk full" in result["error"]
        assert "traceback" in result
        assert "OSError" in result["traceback"]


# ---------------------------------------------------------------------------
# GUI backend captures stderr (TypeScript handler already includes stderr
# when exit code != 0 — verified by reading gui/src/main/handlers/*.ts)
# ---------------------------------------------------------------------------


class TestGuiBackendStderrCapture:
    """Verify the TypeScript handlers capture and surface stderr.

    We can't run the Electron process in pytest, so we verify that:
    1. Bridge scripts that fail with sys.exit(1) still print JSON to stdout
       (so the TypeScript can choose which to use)
    2. The JSON includes the traceback field for CLI consumers
    """

    def test_chat_bridge_prints_traceback_before_exit(self) -> None:
        """chat_bridge prints JSON + traceback to stdout before exiting."""
        import rex_chat_bridge

        with (
            patch("rex_chat_bridge.asyncio") as mock_asyncio,
            patch("sys.exit"),
        ):
            # Make asyncio.run raise the error directly
            mock_asyncio.run.side_effect = RuntimeError("LLM down")

            captured = StringIO()
            with (
                patch.object(sys, "stdin", StringIO('{"message": "hello"}')),
                patch.object(sys, "stdout", captured),
            ):
                rex_chat_bridge.main()

        output = captured.getvalue().strip()
        result = json.loads(output)
        assert result["ok"] is False
        assert "traceback" in result
        assert result["traceback"]
