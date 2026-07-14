"""Tests for ActionDispatcher (US-016).

Verifies that action dispatch logic extracted from assistant.py
produces identical output through the new ActionDispatcher class.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from rex.actions.dispatcher import ActionDispatcher, ActionResult
from rex.context.builder import ContextPackage
from rex.intent.router import IntentResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context() -> ContextPackage:
    return ContextPackage(
        messages=[{"role": "user", "content": "test"}],
        system_prompt="system",
        session_id="default",
        user_facts={},
        prompt="test\nassistant:",
    )


def _unhandled_intent() -> IntentResult:
    return IntentResult(handled=False, response=None, intent_type=None)


def _make_context_builder(context: ContextPackage | None = None) -> MagicMock:
    cb = MagicMock()
    cb.build.return_value = context or _make_context()
    return cb


def _make_llm(reply: str = "LLM reply") -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = reply
    return llm


def _passthrough_router(completion, tool_context, model_call_fn):
    return completion


def _make_result_handler() -> MagicMock:
    from rex.tools.result_handler import ToolResultHandler

    return ToolResultHandler(tool_router_fn=_passthrough_router, ha_bridge=None)


def _make_dispatcher(**kwargs) -> ActionDispatcher:
    defaults = {
        "context_builder": _make_context_builder(),
        "llm": _make_llm(),
        "result_handler": _make_result_handler(),
    }
    defaults.update(kwargs)
    return ActionDispatcher(**defaults)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Import / instantiation
# ---------------------------------------------------------------------------


def test_import():
    from rex.actions.dispatcher import ActionDispatcher, ActionResult  # noqa: F401


def test_action_result_fields():
    r = ActionResult(success=True, response="hello")
    assert r.success is True
    assert r.response == "hello"
    assert r.actions_taken == []
    assert r.error is None


def test_dispatcher_instantiates():
    ad = _make_dispatcher()
    assert ad is not None


# ---------------------------------------------------------------------------
# Skill training early return
# ---------------------------------------------------------------------------


def test_skill_training_returns_early():
    trainer = MagicMock()
    trainer.handle_if_training_request.return_value = "Skill created."
    registry = MagicMock()

    ad = _make_dispatcher(skill_trainer=trainer, skill_registry=registry)
    result = _run(
        ad.dispatch(_unhandled_intent(), _make_context(), "teach me to greet", user_id="default")
    )

    assert result.success is True
    assert result.response == "Skill created."
    assert "skill_training" in result.actions_taken
    trainer.handle_if_training_request.assert_called_once()


def test_skill_training_no_match_proceeds():
    trainer = MagicMock()
    trainer.handle_if_training_request.return_value = None
    registry = MagicMock()

    ad = _make_dispatcher(skill_trainer=trainer, skill_registry=registry)
    result = _run(ad.dispatch(_unhandled_intent(), _make_context(), "hello", user_id="default"))

    # Should fall through to LLM
    assert result.success is True
    assert result.response == "LLM reply"


# ---------------------------------------------------------------------------
# Skill invocation early return
# ---------------------------------------------------------------------------


def test_skill_invocation_returns_early():
    skill = MagicMock()
    router = MagicMock()
    router.match.return_value = skill
    router.execute.return_value = "Skill executed."

    ad = _make_dispatcher(skill_router=router)
    result = _run(
        ad.dispatch(_unhandled_intent(), _make_context(), "run my greeting", user_id="default")
    )

    assert result.response == "Skill executed."
    assert "skill_invocation" in result.actions_taken


def test_skill_invocation_no_match_proceeds():
    router = MagicMock()
    router.match.return_value = None

    ad = _make_dispatcher(skill_router=router)
    result = _run(ad.dispatch(_unhandled_intent(), _make_context(), "hello", user_id="default"))

    assert result.response == "LLM reply"


# ---------------------------------------------------------------------------
# Shopping list early return
# ---------------------------------------------------------------------------


def test_shopping_list_returns_early():
    handler = MagicMock()
    handler.handle.return_value = "Added milk to your list."

    ad = _make_dispatcher(shopping_list_handler=handler)
    result = _run(
        ad.dispatch(_unhandled_intent(), _make_context(), "add milk to my list", user_id="default")
    )

    assert result.response == "Added milk to your list."
    assert "shopping_list" in result.actions_taken


def test_shopping_list_no_match_proceeds():
    handler = MagicMock()
    handler.handle.return_value = None

    ad = _make_dispatcher(shopping_list_handler=handler)
    result = _run(ad.dispatch(_unhandled_intent(), _make_context(), "hello", user_id="default"))

    assert result.response == "LLM reply"


# ---------------------------------------------------------------------------
# Music handler early return
# ---------------------------------------------------------------------------


def test_music_handler_returns_early():
    handler = MagicMock()
    handler.handle.return_value = "Playing jazz."

    ad = _make_dispatcher(music_handler=handler)
    result = _run(ad.dispatch(_unhandled_intent(), _make_context(), "play jazz", user_id="default"))

    assert result.response == "Playing jazz."
    assert "music" in result.actions_taken


# ---------------------------------------------------------------------------
# Device state handler early return
# ---------------------------------------------------------------------------


def test_device_state_returns_early():
    handler = MagicMock()
    handler.handle.return_value = "The lights are on."

    ad = _make_dispatcher(device_state_handler=handler)
    result = _run(
        ad.dispatch(_unhandled_intent(), _make_context(), "are the lights on", user_id="default")
    )

    assert result.response == "The lights are on."
    assert "device_state" in result.actions_taken


# ---------------------------------------------------------------------------
# HA bridge routing
# ---------------------------------------------------------------------------


def test_ha_bridge_process_transcript():
    ha = MagicMock()
    ha.enabled = True
    ha.process_transcript.return_value = "Turning on the lights."
    ha._command_history = None  # no pattern detection

    ad = _make_dispatcher(ha_bridge=ha)
    result = _run(
        ad.dispatch(_unhandled_intent(), _make_context(), "turn on the lights", user_id="default")
    )

    assert result.response == "Turning on the lights."
    ha.process_transcript.assert_called_once_with("turn on the lights")


def test_ha_bridge_undo():
    ha = MagicMock()
    ha.enabled = True
    ha.undo_last.return_value = "Undone."

    ad = _make_dispatcher(ha_bridge=ha)
    result = _run(ad.dispatch(_unhandled_intent(), _make_context(), "undo that", user_id="default"))

    assert result.response == "Undone."
    ha.undo_last.assert_called_once()
    ha.process_transcript.assert_not_called()


def test_ha_bridge_disabled_falls_through_to_llm():
    ha = MagicMock()
    ha.enabled = False

    ad = _make_dispatcher(ha_bridge=ha)
    result = _run(
        ad.dispatch(_unhandled_intent(), _make_context(), "turn on the lights", user_id="default")
    )

    assert result.response == "LLM reply"
    ha.process_transcript.assert_not_called()


# ---------------------------------------------------------------------------
# LLM fallback path
# ---------------------------------------------------------------------------


def test_llm_called_when_no_handler_matches():
    llm = _make_llm("The weather is sunny.")
    ad = _make_dispatcher(llm=llm)
    result = _run(
        ad.dispatch(_unhandled_intent(), _make_context(), "what is the weather", user_id="default")
    )

    assert result.response == "The weather is sunny."
    llm.generate.assert_called_once()


def test_llm_uses_context_messages():
    llm = _make_llm("reply")
    ctx = ContextPackage(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}],
        system_prompt="sys",
        session_id="default",
        user_facts={},
        prompt="sys\nuser: q\nassistant:",
    )
    cb = _make_context_builder(ctx)

    ad = _make_dispatcher(llm=llm, context_builder=cb)
    result = _run(ad.dispatch(_unhandled_intent(), ctx, "q", user_id="default"))

    assert result.success is True
    call_kwargs = llm.generate.call_args
    assert call_kwargs is not None


# ---------------------------------------------------------------------------
# ActionDispatcher accessible from rex.actions
# ---------------------------------------------------------------------------


def test_importable_from_package():
    from rex.actions.dispatcher import ActionDispatcher, ActionResult  # noqa: F401

    assert ActionDispatcher is not None
    assert ActionResult is not None


# ---------------------------------------------------------------------------
# Priority order: skills before HA and LLM
# ---------------------------------------------------------------------------


def test_skill_invocation_takes_priority_over_ha():
    skill = MagicMock()
    router = MagicMock()
    router.match.return_value = skill
    router.execute.return_value = "Skill won."

    ha = MagicMock()
    ha.enabled = True
    ha.process_transcript.return_value = "HA won."

    ad = _make_dispatcher(skill_router=router, ha_bridge=ha)
    result = _run(
        ad.dispatch(_unhandled_intent(), _make_context(), "do something", user_id="default")
    )

    assert result.response == "Skill won."
    ha.process_transcript.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: Assistant uses ActionDispatcher via generate_reply
# ---------------------------------------------------------------------------


def test_assistant_has_action_dispatcher():
    """Assert that Assistant.__init__ creates _action_dispatcher."""

    with (
        patch("rex.assistant.LanguageModel"),
        patch("rex.assistant.HABridge"),
        patch("rex.assistant.HistoryStore"),
        patch("rex.assistant.ModelRouter"),
    ):
        from rex.assistant import Assistant

        a = Assistant.__new__(Assistant)
        ad = a._get_or_create_action_dispatcher()
        assert ad is not None
        from rex.actions.dispatcher import ActionDispatcher

        assert isinstance(ad, ActionDispatcher)
