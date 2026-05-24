"""Tests for ToolResultHandler (US-013).

Verifies that tool result post-processing extracted from assistant.py
produces identical output through the new ToolResultHandler class.
"""

from __future__ import annotations

import asyncio

from rex.tools.result_handler import ToolResultHandler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _passthrough_router(completion, tool_context, model_call_fn):
    """Router that returns the completion unchanged (no TOOL_REQUEST found)."""
    return completion


def _make_handler(ha_bridge=None):
    return ToolResultHandler(tool_router_fn=_passthrough_router, ha_bridge=ha_bridge)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_import():
    """ToolResultHandler is importable from rex.tools.result_handler."""
    from rex.tools.result_handler import ToolResultHandler  # noqa: F401


def test_process_passthrough():
    """Clean completion passes through unchanged."""
    handler = _make_handler()
    result = asyncio.run(
        handler.process(
            "hello",
            "Here is some information.",
            tool_context={},
            model_call_fn=None,
            plugin_enrichments=[],
        )
    )
    assert result == "Here is some information."


def test_process_appends_plugin_enrichments():
    """Plugin enrichment strings are appended to the completion."""
    handler = _make_handler()
    result = asyncio.run(
        handler.process(
            "hello",
            "Base reply.",
            tool_context={},
            model_call_fn=None,
            plugin_enrichments=["extra fact"],
        )
    )
    assert "Base reply." in result
    assert "extra fact" in result
    assert "Additional info:" in result


def test_process_multiple_enrichments():
    """Multiple plugin enrichments are all appended."""
    handler = _make_handler()
    result = asyncio.run(
        handler.process(
            "hello",
            "Base reply.",
            tool_context={},
            model_call_fn=None,
            plugin_enrichments=["fact one", "fact two"],
        )
    )
    assert "fact one" in result
    assert "fact two" in result


def test_guard_suppresses_unverified_claim():
    """Unverified action claim is replaced with a safe fallback message."""
    handler = _make_handler()
    result = asyncio.run(
        handler.process(
            "what is the weather",
            "I have added that to your calendar.",
            tool_context={},
            model_call_fn=None,
            plugin_enrichments=[],
        )
    )
    assert "I did not change anything" in result
    assert "calendar" not in result


def test_guard_allows_explicit_mutation_request():
    """When the transcript is an explicit mutation request, action claims pass through."""
    handler = _make_handler()
    result = asyncio.run(
        handler.process(
            "add a meeting to my calendar",
            "I have added that to your calendar.",
            tool_context={},
            model_call_fn=None,
            plugin_enrichments=[],
        )
    )
    assert "I have added that to your calendar." in result


def test_guard_recipe_fallback():
    """Unverified action claim on a chocolate cake recipe request returns the recipe."""
    handler = _make_handler()
    result = asyncio.run(
        handler.process(
            "how do I make a chocolate cake",
            "I have added the chocolate cake recipe to your notes.",
            tool_context={},
            model_call_fn=None,
            plugin_enrichments=[],
        )
    )
    assert "chocolate cake recipe" in result.lower()
    assert "flour" in result


def test_contains_internal_tool_syntax_detection():
    """_contains_internal_tool_syntax correctly detects TOOL_REQUEST markers."""
    handler = _make_handler()
    assert handler._contains_internal_tool_syntax('TOOL_REQUEST: {"tool": "time_now"}')
    assert handler._contains_internal_tool_syntax("some text TOOL_RESULT: foo")
    assert not handler._contains_internal_tool_syntax("normal reply text")


def test_sanitize_resolves_tool_syntax():
    """_sanitize_internal_tool_output resolves directives when router returns clean text."""
    resolved_text = "It is 3:00 PM."

    def resolving_router(completion, tool_context, model_call_fn):
        return resolved_text

    handler = ToolResultHandler(tool_router_fn=resolving_router)
    result = handler._sanitize_internal_tool_output(
        "what time is it",
        'TOOL_REQUEST: {"tool": "time_now"}',
        {},
        None,
    )
    assert result == resolved_text


def test_sanitize_suppresses_when_still_contains_syntax():
    """_sanitize_internal_tool_output returns fallback when router leaves tool syntax."""

    def bad_router(completion, tool_context, model_call_fn):
        return "TOOL_REQUEST: still here"

    handler = ToolResultHandler(tool_router_fn=bad_router)
    result = handler._sanitize_internal_tool_output(
        "what time is it",
        'TOOL_REQUEST: {"tool": "time_now"}',
        {},
        None,
    )
    assert result == "I could not complete that tool request."


def test_no_ha_bridge_skips_ha_processing():
    """Handler with no HA bridge skips HA post-processing without error."""
    handler = _make_handler(ha_bridge=None)
    result = asyncio.run(
        handler.process(
            "turn on the lights",
            "Lights turned on.",
            tool_context={},
            model_call_fn=None,
            plugin_enrichments=[],
        )
    )
    assert result == "Lights turned on."


def test_ha_bridge_post_processes_when_enabled():
    """HA bridge post_process_response is called when bridge is enabled."""

    class FakeHABridge:
        enabled = True

        def post_process_response(self, completion):
            return f"[HA] {completion}"

    handler = ToolResultHandler(
        tool_router_fn=_passthrough_router,
        ha_bridge=FakeHABridge(),
    )
    result = asyncio.run(
        handler.process(
            "turn on the lights",
            "Done.",
            tool_context={},
            model_call_fn=None,
            plugin_enrichments=[],
        )
    )
    assert result == "[HA] Done."


def test_ha_bridge_skipped_when_disabled():
    """HA bridge post_process_response is NOT called when bridge.enabled is False."""

    class FakeHABridge:
        enabled = False

        def post_process_response(self, completion):
            raise AssertionError("Should not be called when disabled")

    handler = ToolResultHandler(
        tool_router_fn=_passthrough_router,
        ha_bridge=FakeHABridge(),
    )
    result = asyncio.run(
        handler.process(
            "hello",
            "Hi there.",
            tool_context={},
            model_call_fn=None,
            plugin_enrichments=[],
        )
    )
    assert result == "Hi there."
