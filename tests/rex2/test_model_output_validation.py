from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rex.actions.dispatcher import ActionResult
from rex.assistant import Assistant
from rex.intent.router import IntentResult
from rex.response.builder import FinalResponse
from rex.runtime.events import EventKind

SAFE_FAILURE = (
    "I couldn't produce a reliable response from the selected model. " "Please try again."
)


def _assistant() -> Assistant:
    assistant = Assistant.__new__(Assistant)
    assistant._settings = SimpleNamespace(
        max_memory_items=50,
        persist_history=False,
        followups_enabled=False,
        model_routing=None,
        transcripts_enabled=False,
        llm_provider="test-provider",
        llm_model="test-model",
        llm=None,
    )
    assistant._user_id = "james"
    assistant._histories = {}
    assistant._history_limit = 50
    assistant._plugins = []
    assistant._history_store = None
    assistant._followup_engine = None
    assistant._followup_sessions = set()
    assistant._followup_bootstrap_pending = False
    assistant._pending_followups = {}
    assistant._response_cache = None
    assistant._ha_bridge = None
    assistant._suggestion_engine = None
    assistant._pattern_entries = {}
    assistant._router = None
    assistant._llm = MagicMock()
    assistant._llm.model_name = "test-model"
    assistant._context_builder = MagicMock()
    assistant._context_builder.build.return_value = SimpleNamespace(
        messages=[], prompt="prompt", system_prompt="system"
    )
    assistant._response_builder = MagicMock()
    assistant._response_builder.check_cache.return_value = None
    assistant._response_builder.build.side_effect = lambda result, _ctx, **_kwargs: FinalResponse(
        text=result.response,
        tts_text=result.response,
    )
    assistant._turn_events = []
    assistant._turn_event_observer = assistant._turn_events.append
    assistant._log_turn = MagicMock()
    intent_router = MagicMock()
    intent_router.route.return_value = IntentResult(handled=False, response=None, intent_type=None)
    assistant._intent_router = intent_router
    return assistant


def _set_dispatch_response(assistant: Assistant, response: str) -> None:
    async def dispatch(*_args, turn_events=None, **_kwargs):
        if turn_events is not None:
            turn_events.emit(
                EventKind.MODEL_PROGRESS,
                {"stage": "generation", "status": "returned"},
            )
        return ActionResult(
            success=True,
            response=response,
            actions_taken=["llm"],
            model_generated=True,
        )

    assistant._action_dispatcher = MagicMock()
    assistant._action_dispatcher.dispatch = dispatch


async def _collect(assistant: Assistant, text: str, *, voice_mode: bool = False) -> list[str]:
    return [
        chunk
        async for chunk in assistant.stream_reply(
            text,
            voice_mode=voice_mode,
        )
    ]


def _repetition_flood() -> str:
    return " ".join(["broken-loop"] * 240)


def test_generate_reply_replaces_obvious_repetition_with_model_failure() -> None:
    assistant = _assistant()
    bad_output = _repetition_flood()
    _set_dispatch_response(assistant, bad_output)

    reply = asyncio.run(assistant.generate_reply("explain this"))

    assert reply == SAFE_FAILURE
    assert bad_output not in reply
    assistant._response_builder.build.assert_not_called()
    assert assistant._turn_events[-1].kind is EventKind.FAILED
    assert assistant._turn_events[-1].details.get("failure_kind") == "model_failure"


def test_stream_and_voice_use_same_model_failure_guard() -> None:
    assistant = _assistant()
    _set_dispatch_response(assistant, _repetition_flood())

    text_chunks = asyncio.run(_collect(assistant, "stream request"))
    assistant._turn_events.clear()
    voice_chunks = asyncio.run(_collect(assistant, "voice request", voice_mode=True))

    assert " ".join(text_chunks) == SAFE_FAILURE
    assert " ".join(voice_chunks) == SAFE_FAILURE
    assert assistant._turn_events[-1].kind is EventKind.FAILED


def test_model_failure_log_is_diagnostic_without_response_content(caplog) -> None:
    assistant = _assistant()
    bad_output = _repetition_flood()
    _set_dispatch_response(assistant, bad_output)

    with caplog.at_level(logging.WARNING, logger="rex.assistant"):
        reply = asyncio.run(assistant.generate_reply("diagnose"))

    assert reply == SAFE_FAILURE
    assert "provider=test-provider" in caplog.text
    assert "model=test-model" in caplog.text
    assert "route=action_dispatch" in caplog.text
    assert f"output_length={len(bad_output)}" in caplog.text
    assert "reason=repeated_token_flood" in caplog.text
    assert bad_output not in caplog.text


def test_routed_model_output_is_validated_before_terminal_response(caplog) -> None:
    assistant = _assistant()
    assistant._llm.model_name = "fast-model"
    router = MagicMock()
    router.classify.return_value = "complex"
    router.resolve_model.return_value = "deep-model"
    assistant._router = router
    _set_dispatch_response(assistant, _repetition_flood())

    with caplog.at_level(logging.WARNING, logger="rex.assistant"):
        reply = asyncio.run(assistant.generate_reply("complex request"))

    assert reply == SAFE_FAILURE
    assert "model=deep-model" in caplog.text
    assert assistant._llm.model_name == "fast-model"
    assert assistant._turn_events[-1].kind is EventKind.FAILED


def test_normal_answer_and_refusal_are_not_reclassified() -> None:
    assistant = _assistant()
    normal = "I can't help with that request, but I can explain the underlying concept."
    _set_dispatch_response(assistant, normal)

    reply = asyncio.run(assistant.generate_reply("normal request"))

    assert reply == normal
    assert assistant._turn_events[-1].kind is EventKind.COMPLETED


@pytest.mark.parametrize(
    "bad_output",
    [
        'Error code: 429 - {"error": {"message": "rate limit exceeded"}}',
        "[Ollama: connection failed — is Ollama running?]",
        "[Ollama: model 'qwen' not found — run: ollama pull qwen]",
        "[Ollama: unexpected error: provider exploded]",
        "(silence)",
        "!@#$%^&*()_+{}[]<>?/|\\" * 120,
    ],
)
def test_other_obvious_provider_failure_shapes_are_replaced(bad_output: str) -> None:
    assistant = _assistant()
    _set_dispatch_response(assistant, bad_output)

    reply = asyncio.run(assistant.generate_reply("provider request"))

    assert reply == SAFE_FAILURE
    assert bad_output not in reply
    assert assistant._turn_events[-1].kind is EventKind.FAILED


def test_legitimate_json_error_example_is_not_reclassified() -> None:
    assistant = _assistant()
    example = '{"error":"invalid input","code":400}'
    _set_dispatch_response(assistant, example)

    reply = asyncio.run(assistant.generate_reply("show me a JSON error example"))

    assert reply == example
    assert assistant._turn_events[-1].kind is EventKind.COMPLETED


def test_bounded_repetition_and_symbolic_output_are_not_reclassified() -> None:
    from rex.response.validation import validate_model_output

    examples = [
        "-" * 200,
        " ".join(["test"] * 100),
        "😀" * 400,
        "separator\n" * 10,
    ]

    assert all(validate_model_output(example).valid for example in examples)


def test_unicode_repetition_flood_is_rejected() -> None:
    from rex.response.validation import validate_model_output

    result = validate_model_output(" ".join(["сломано"] * 240))

    assert result.valid is False
    assert result.reason == "repeated_token_flood"


def test_real_dispatcher_marks_deterministic_skill_response_as_non_model() -> None:
    from rex.actions.dispatcher import ActionDispatcher

    assistant = _assistant()
    trainer = MagicMock()
    trainer.handle_if_training_request.return_value = _repetition_flood()
    dispatcher = ActionDispatcher(
        context_builder=assistant._context_builder,
        llm=assistant._llm,
        result_handler=MagicMock(),
        skill_trainer=trainer,
        skill_registry=MagicMock(),
    )
    assistant._action_dispatcher = dispatcher

    reply = asyncio.run(assistant.generate_reply("teach a deterministic skill"))

    assert reply == _repetition_flood()
    assert assistant._turn_events[-1].kind is EventKind.COMPLETED
    assistant._llm.generate.assert_not_called()


def test_real_dispatcher_validates_model_tool_reprompt_before_terminal_response() -> None:
    from rex.actions.dispatcher import ActionDispatcher

    assistant = _assistant()
    assistant._context_builder.build_system_context.return_value = "system"
    assistant._llm.generate.side_effect = ["tool-request", _repetition_flood()]

    class RePromptingResultHandler:
        async def process(
            self,
            _transcript,
            _completion,
            *,
            tool_context,
            model_call_fn,
            plugin_enrichments,
        ):
            del tool_context, plugin_enrichments
            return model_call_fn({"role": "tool", "content": "verified tool result"})

    assistant._action_dispatcher = ActionDispatcher(
        context_builder=assistant._context_builder,
        llm=assistant._llm,
        result_handler=RePromptingResultHandler(),
        model_call_fn_builder=assistant._build_tool_model_call,
    )

    reply = asyncio.run(assistant.generate_reply("use a tool and summarize it"))

    assert reply == SAFE_FAILURE
    assert assistant._llm.generate.call_count == 2
    assert assistant._response_builder.build.call_count == 0
    assert assistant._turn_events[-1].kind is EventKind.FAILED
    assert assistant._turn_events[-1].details.get("failure_kind") == "model_failure"
