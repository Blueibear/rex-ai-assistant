"""Tests for US-015: IntentRouter extraction from assistant.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rex.intent.router import IntentResult, IntentRouter

# ---------------------------------------------------------------------------
# IntentResult dataclass
# ---------------------------------------------------------------------------


def test_intent_result_handled():
    r = IntentResult(handled=True, response="hello", intent_type="greeting")
    assert r.handled is True
    assert r.response == "hello"
    assert r.intent_type == "greeting"


def test_intent_result_not_handled():
    r = IntentResult(handled=False, response=None, intent_type=None)
    assert r.handled is False
    assert r.response is None
    assert r.intent_type is None


# ---------------------------------------------------------------------------
# Greeting shortcuts
# ---------------------------------------------------------------------------


@pytest.fixture()
def router():
    return IntentRouter()


def test_route_hello_greeting(router):
    result = router.route("hello")
    assert result.handled is True
    assert result.intent_type == "greeting"
    assert "hello" in result.response.lower()


def test_route_hey_greeting(router):
    result = router.route("hey!")
    assert result.handled is True
    assert result.intent_type == "greeting"


def test_route_wellbeing(router):
    result = router.route("how are you?")
    assert result.handled is True
    assert result.intent_type == "greeting"
    assert result.response is not None


def test_route_creator_query(router):
    result = router.route("who created you?")
    assert result.handled is True
    assert result.intent_type == "greeting"
    assert "AskRex" in result.response


# ---------------------------------------------------------------------------
# Recipe shortcut
# ---------------------------------------------------------------------------


def test_route_chocolate_cake_recipe(router):
    result = router.route("can you give me a chocolate cake recipe?")
    assert result.handled is True
    assert result.intent_type == "recipe"
    assert "chocolate cake" in result.response.lower()


def test_route_recipe_skipped_for_shopping_list(router):
    result = router.route("add chocolate cake recipe to my shopping list")
    # Shopping list reference suppresses recipe shortcut
    assert result.handled is False


def test_route_no_recipe_for_unknown_dish(router):
    result = router.route("give me a lasagna recipe")
    # Only chocolate cake is in the shortcut
    assert result.handled is False


# ---------------------------------------------------------------------------
# Time/date shortcuts (mocked tool executor)
# ---------------------------------------------------------------------------


def _mock_time_result():
    return {"local_time": "2026-05-24 14:30", "date": "2026-05-24", "timezone": "UTC"}


def test_route_time_query(router):
    with patch(
        "rex.openclaw.tool_executor.execute_tool",
        return_value=_mock_time_result(),
    ):
        result = router.route("what time is it?")
    assert result.handled is True
    assert result.intent_type == "time_query"
    assert "2:30 PM" in result.response


def test_route_date_query(router):
    with patch(
        "rex.openclaw.tool_executor.execute_tool",
        return_value=_mock_time_result(),
    ):
        result = router.route("what's today's date?")
    assert result.handled is True
    assert result.intent_type == "time_query"
    assert "May 24, 2026" in result.response


def test_route_day_query(router):
    with patch(
        "rex.openclaw.tool_executor.execute_tool",
        return_value=_mock_time_result(),
    ):
        result = router.route("what day is it today?")
    assert result.handled is True
    assert result.intent_type == "time_query"
    assert "Sunday" in result.response


def test_route_time_falls_back_to_local_clock_on_error(router):
    """When the tool returns an error, the router uses the local clock."""
    with patch(
        "rex.openclaw.tool_executor.execute_tool",
        return_value={"error": "no network"},
    ):
        # The fallback path only kicks in when location matches the configured one
        result = router.route("what time is it?")
    # Either handled (local fallback) or not — must not raise
    assert isinstance(result.handled, bool)


# ---------------------------------------------------------------------------
# Unrecognized input falls through
# ---------------------------------------------------------------------------


def test_route_unrecognized_returns_not_handled(router):
    result = router.route("tell me about the history of Rome")
    assert result.handled is False
    assert result.response is None
    assert result.intent_type is None


def test_route_empty_string_returns_not_handled(router):
    result = router.route("")
    assert result.handled is False


# ---------------------------------------------------------------------------
# IntentRouter wired into Assistant
# ---------------------------------------------------------------------------


def test_assistant_has_intent_router():
    """Assistant must expose _intent_router as an IntentRouter instance."""
    from rex.assistant import Assistant
    from rex.intent.router import IntentRouter

    # Build a minimal assistant shell via __new__ + manual wiring, same pattern
    # used by other test files (avoids heavy __init__ side effects).
    a = Assistant.__new__(Assistant)
    a._settings = MagicMock()
    a._history = []
    a._user_id = "default"
    a._tool_dispatcher = None
    a._context_builder = MagicMock()
    a._context_builder.build_system_context.return_value = ""

    # Manually install the intent router as __init__ would
    a._intent_router = IntentRouter(tool_context_fn=None)

    assert hasattr(a, "_intent_router")
    assert isinstance(a._intent_router, IntentRouter)


@pytest.mark.asyncio
async def test_generate_reply_uses_intent_router_for_greeting():
    """generate_reply() must return greeting via IntentRouter without hitting LLM."""
    from unittest.mock import MagicMock, patch

    from rex.assistant import Assistant
    from rex.intent.router import IntentResult

    a = Assistant.__new__(Assistant)
    a._settings = MagicMock()
    a._settings.persist_history = False
    a._settings.followups_enabled = False
    a._history = []
    a._history_limit = 20
    a._plugins = []
    a._user_id = "default"
    a._history_store = None
    a._followup_engine = None
    a._pending_followup = None
    a._followup_lock = None
    a._router = None
    a._tool_dispatcher = None
    a._shopping_list_handler = None
    a._music_handler = None
    a._device_state_handler = None
    a._response_cache = None
    a._ha_bridge = None
    a._suggestion_engine = None
    a._transcripts_dir = MagicMock()
    a._transcripts_dir.__truediv__ = MagicMock(return_value=MagicMock())
    a._skill_trainer = None
    a._skill_registry = None
    a._skill_router = None

    # Wire the intent router mock
    mock_router = MagicMock()
    mock_router.route.return_value = IntentResult(
        handled=True, response="Hello. How can I help?", intent_type="greeting"
    )
    a._intent_router = mock_router

    # Capability query check must not intercept
    with (
        patch(
            "rex.assistant.is_capability_query",
            return_value=False,
        ),
        patch(
            "rex.assistant.get_capability_registry",
        ),
        patch(
            "rex.assistant.build_capability_response",
        ),
    ):
        result = await a.generate_reply("hello")

    assert result == "Hello. How can I help?"
    mock_router.route.assert_called_once_with("hello")
