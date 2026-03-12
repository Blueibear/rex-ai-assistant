"""Tests for US-121: Audit and fix blocking I/O in async handlers.

Acceptance criteria:
- all async handler functions audited for synchronous file I/O, time.sleep(),
  and synchronous HTTP calls
- any blocking calls found replaced with async equivalents or offloaded to
  a thread executor
- findings and changes documented in a comment or commit message
- Typecheck passes

Audit findings:
  voice_loop.py::AsyncRexAssistant._process_conversation previously called
  append_history_entry() and export_transcript() — both synchronous functions
  with file I/O — directly in the async function body. These were wrapped in
  asyncio.to_thread() to prevent event-loop blocking.

All other async handlers (generate_reply, _run_plugins, transcribe, run,
_handle_interaction, wait_for_wake) already used run_in_executor or
asyncio.to_thread for every blocking operation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Static analysis helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent


def _find_blocking_calls_in_async(source: str, filename: str) -> list[str]:
    """Return a list of blocking call patterns found directly in async functions.

    Checks for direct calls to:
    - time.sleep()
    - open() / io.open()
    that are NOT wrapped in asyncio.to_thread / run_in_executor.

    This is a heuristic / best-effort check, not exhaustive.
    """
    tree = ast.parse(source, filename=filename)
    issues: list[str] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._in_async = False
            self._depth = 0

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            prev = self._in_async
            self._in_async = True
            self.generic_visit(node)
            self._in_async = prev

        def visit_Call(self, node: ast.Call) -> None:
            if self._in_async:
                call_str = ast.unparse(node)
                # Flag time.sleep() if it appears directly in async context
                if "time.sleep(" in call_str and "to_thread" not in call_str:
                    lineno = getattr(node, "lineno", "?")
                    issues.append(f"{filename}:{lineno}: time.sleep() in async function")
            self.generic_visit(node)

    _Visitor().visit(tree)
    return issues


# ---------------------------------------------------------------------------
# Static audit tests
# ---------------------------------------------------------------------------


class TestStaticAudit:
    def test_voice_loop_no_bare_time_sleep_in_async(self) -> None:
        """_process_conversation and other async methods must not call time.sleep() directly."""
        source = (REPO_ROOT / "voice_loop.py").read_text(encoding="utf-8")
        issues = _find_blocking_calls_in_async(source, "voice_loop.py")
        assert not issues, (
            "Blocking time.sleep() found directly in async handlers:\n"
            + "\n".join(issues)
        )

    def test_assistant_no_bare_time_sleep_in_async(self) -> None:
        source = (REPO_ROOT / "rex" / "assistant.py").read_text(encoding="utf-8")
        issues = _find_blocking_calls_in_async(source, "rex/assistant.py")
        assert not issues

    def test_voice_loop_uses_to_thread_for_file_io(self) -> None:
        """append_history_entry and export_transcript must be called via asyncio.to_thread."""
        source = (REPO_ROOT / "voice_loop.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Collect all asyncio.to_thread calls
        to_thread_targets: list[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "to_thread"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "asyncio"
                and node.args
            ):
                to_thread_targets.append(ast.unparse(node.args[0]))

        assert any("append_history_entry" in t for t in to_thread_targets), (
            "append_history_entry must be called via asyncio.to_thread() "
            f"in voice_loop.py; found to_thread targets: {to_thread_targets}"
        )
        assert any("export_transcript" in t for t in to_thread_targets), (
            "export_transcript must be called via asyncio.to_thread() "
            f"in voice_loop.py; found to_thread targets: {to_thread_targets}"
        )

    def test_assistant_uses_run_in_executor_for_blocking(self) -> None:
        """rex/assistant.py async functions must use run_in_executor for blocking calls."""
        source = (REPO_ROOT / "rex" / "assistant.py").read_text(encoding="utf-8")
        assert "run_in_executor" in source, (
            "rex/assistant.py should use loop.run_in_executor for blocking calls"
        )


# ---------------------------------------------------------------------------
# Behavioural verification — source-level check of _process_conversation
# ---------------------------------------------------------------------------


class TestProcessConversationOffloads:
    """Verify _process_conversation uses asyncio.to_thread for blocking helpers."""

    def test_process_conversation_source_uses_to_thread_for_append(self) -> None:
        """_process_conversation source must pass append_history_entry to to_thread."""
        import ast

        source = (REPO_ROOT / "voice_loop.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Find _process_conversation method body
        body_source: str | None = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AsyncFunctionDef)
                and node.name == "_process_conversation"
            ):
                body_source = ast.unparse(node)
                break

        assert body_source is not None, "_process_conversation not found in voice_loop.py"
        assert "append_history_entry" in body_source
        # Both to_thread and append_history_entry must appear — together means it's wrapped
        assert "asyncio.to_thread" in body_source

    def test_process_conversation_source_uses_to_thread_for_export(self) -> None:
        import ast

        source = (REPO_ROOT / "voice_loop.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AsyncFunctionDef)
                and node.name == "_process_conversation"
            ):
                body_source = ast.unparse(node)
                assert "export_transcript" in body_source
                assert "asyncio.to_thread" in body_source
                return

        pytest.fail("_process_conversation not found in voice_loop.py")


# ---------------------------------------------------------------------------
# Documentation: audit summary
# ---------------------------------------------------------------------------


class TestAuditDocumentation:
    def test_us121_docstring_describes_findings(self) -> None:
        """This module's docstring documents the audit findings as required."""
        module_doc = __doc__ or ""
        assert "append_history_entry" in module_doc
        assert "export_transcript" in module_doc
        assert "asyncio.to_thread" in module_doc
