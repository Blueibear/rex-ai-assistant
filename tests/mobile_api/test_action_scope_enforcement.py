"""Lower-layer mobile capability enforcement tests (S6)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from rex.mobile_api.action_context import (
    MobileActionDeniedError,
    mobile_action_context,
)
from rex.tools.dispatcher import ToolDispatcher
from rex.tools.registry import Tool, ToolRegistry
from tests.test_us016_action_dispatcher import (
    _make_context,
    _make_dispatcher,
    _unhandled_intent,
)


def _tool(name: str, tags: list[str], operation: str = "read") -> tuple[Tool, MagicMock]:
    handler = MagicMock(return_value={"ok": True})
    return (
        Tool(
            name=name,
            description=name,
            capability_tags=tags,
            requires_config=[],
            handler=handler,
            operation=operation,
        ),
        handler,
    )


def test_tool_dispatcher_allows_safe_chat_tool_with_chat_scope() -> None:
    tool, handler = _tool("weather_now", ["weather"], "read")
    registry = ToolRegistry()
    registry.register(tool)
    with mobile_action_context(frozenset({"chat.send"})):
        result = ToolDispatcher(registry).dispatch("weather_now", {}, {})
    assert result.success is True
    handler.assert_called_once()


def test_tool_dispatcher_denies_home_mutation_without_home_scope() -> None:
    tool, handler = _tool("home_assistant_call_service", ["smart_home"], "mutation")
    registry = ToolRegistry()
    registry.register(tool)
    with mobile_action_context(frozenset({"chat.send"})):
        with pytest.raises(MobileActionDeniedError):
            ToolDispatcher(registry).dispatch("home_assistant_call_service", {}, {})
    handler.assert_not_called()


def test_action_dispatcher_propagates_mobile_context_into_executor_thread() -> None:
    tool, handler = _tool("home_assistant_call_service", ["smart_home"], "mutation")
    registry = ToolRegistry()
    registry.register(tool)
    dispatcher = _make_dispatcher(tool_dispatcher=ToolDispatcher(registry))
    with mobile_action_context(frozenset({"chat.send"})):
        with pytest.raises(MobileActionDeniedError):
            asyncio.run(
                dispatcher.dispatch(
                    _unhandled_intent(),
                    _make_context(),
                    "turn on the lights",
                    user_id="default",
                )
            )
    handler.assert_not_called()


def test_direct_ha_bridge_runs_only_with_home_control() -> None:
    ha = MagicMock()
    ha.enabled = True
    ha.process_transcript.return_value = "Lights changed."
    ha._command_history = None
    dispatcher = _make_dispatcher(ha_bridge=ha)

    with mobile_action_context(frozenset({"chat.send"})):
        denied_result = asyncio.run(
            dispatcher.dispatch(
                _unhandled_intent(),
                _make_context(),
                "turn on the lights",
                user_id="default",
            )
        )
    assert denied_result.response == "LLM reply"
    ha.process_transcript.assert_not_called()

    with mobile_action_context(
        frozenset({"chat.send", "home.control"}),
        permissions=frozenset({"ha_control"}),
    ):
        allowed_result = asyncio.run(
            dispatcher.dispatch(
                _unhandled_intent(),
                _make_context(),
                "turn on the lights",
                user_id="default",
            )
        )
    assert allowed_result.response == "Lights changed."
    ha.process_transcript.assert_called_once_with("turn on the lights")


def test_live_revalidation_runs_before_lower_action() -> None:
    calls: list[str] = []
    tool, handler = _tool("weather_now", ["weather"], "read")
    registry = ToolRegistry()
    registry.register(tool)

    def revoked() -> None:
        calls.append("checked")
        raise RuntimeError("revoked")

    with mobile_action_context(frozenset({"chat.send"}), revalidate=revoked):
        with pytest.raises(RuntimeError, match="revoked"):
            ToolDispatcher(registry).dispatch("weather_now", {}, {})
    assert calls == ["checked"]
    handler.assert_not_called()


def test_unmapped_local_tool_is_denied_for_mobile() -> None:
    from rex.local_tool_executor import execute_tool

    with mobile_action_context(frozenset({"chat.send"})):
        with pytest.raises(MobileActionDeniedError):
            execute_tool(
                "send_email",
                {"to": "nobody@example.test", "body": "test", "_user_id": "default"},
            )


def test_ha_response_post_processing_requires_home_control_scope() -> None:
    from rex.tools.result_handler import ToolResultHandler

    ha = MagicMock()
    ha.enabled = True
    ha.post_process_response.return_value = "Lights changed."
    handler = ToolResultHandler(tool_router_fn=lambda completion, *_args: completion, ha_bridge=ha)

    with mobile_action_context(frozenset({"chat.send"})):
        denied = asyncio.run(
            handler.process(
                "turn on the lights",
                "[[ha:light.turn_on entity_id=light.office]]",
                tool_context={},
                model_call_fn=None,
                plugin_enrichments=[],
            )
        )
    assert denied == "[[ha:light.turn_on entity_id=light.office]]"
    ha.post_process_response.assert_not_called()

    with mobile_action_context(
        frozenset({"chat.send", "home.control"}),
        permissions=frozenset({"ha_control"}),
    ):
        allowed = asyncio.run(
            handler.process(
                "turn on the lights",
                "[[ha:light.turn_on entity_id=light.office]]",
                tool_context={},
                model_call_fn=None,
                plugin_enrichments=[],
            )
        )
    assert allowed == "Lights changed."
    ha.post_process_response.assert_called_once()


def test_tool_name_fragment_cannot_invent_task_scope() -> None:
    tool, handler = _tool("task_exfiltrate", ["untrusted"], "mutation")
    registry = ToolRegistry()
    registry.register(tool)
    with mobile_action_context(frozenset({"tasks.write"})):
        with pytest.raises(MobileActionDeniedError):
            ToolDispatcher(registry).dispatch("task_exfiltrate", {}, {})
    handler.assert_not_called()


def test_dynamic_openclaw_tool_is_denied_for_mobile_even_with_chat_scope() -> None:
    from rex.openclaw.tool_executor import execute_tool

    with mobile_action_context(frozenset({"chat.send"})):
        with pytest.raises(MobileActionDeniedError):
            execute_tool(
                {"tool": "weather_now", "args": {}},
                {},
                skip_policy_check=True,
                skip_audit_log=True,
            )


def test_skill_training_remains_desktop_only_even_with_tasks_write() -> None:
    trainer = MagicMock()
    trainer.handle_if_training_request.return_value = "Skill created."
    registry = MagicMock()
    dispatcher = _make_dispatcher(skill_trainer=trainer, skill_registry=registry)

    with mobile_action_context(frozenset({"chat.send", "tasks.write"})):
        result = asyncio.run(
            dispatcher.dispatch(
                _unhandled_intent(),
                _make_context(),
                "teach Rex a new skill",
                user_id="default",
            )
        )
    assert result.response == "LLM reply"
    trainer.handle_if_training_request.assert_not_called()


def test_home_scope_alone_cannot_bypass_live_user_permission() -> None:
    tool, handler = _tool("home_assistant_call_service", ["smart_home"], "mutation")
    registry = ToolRegistry()
    registry.register(tool)

    with mobile_action_context(frozenset({"home.control"})):
        with pytest.raises(MobileActionDeniedError):
            ToolDispatcher(registry).dispatch("home_assistant_call_service", {}, {})
    handler.assert_not_called()

    with mobile_action_context(frozenset({"home.control"}), permissions=frozenset({"admin"})):
        result = ToolDispatcher(registry).dispatch(
            "home_assistant_call_service", {}, {"user_id": "default"}
        )
    assert result.status == "attempted_unverified"
    handler.assert_called_once()
