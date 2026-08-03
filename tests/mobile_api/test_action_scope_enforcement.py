"""Lower-layer mobile capability enforcement tests (S6)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from rex.mobile_api.action_context import (
    MobileActionDeniedError,
    MobileStrongAuthRequiredError,
    authorized_mobile_tool,
    mobile_action_context,
)
from rex.mobile_api.strong_auth import StrongAuthError
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
    tool, handler = _tool("weather_now", ["weather"], "read")
    registry = ToolRegistry()
    registry.register(tool)
    dispatcher = _make_dispatcher(tool_dispatcher=ToolDispatcher(registry))
    with mobile_action_context(frozenset()):
        with pytest.raises(MobileActionDeniedError):
            asyncio.run(
                dispatcher.dispatch(
                    _unhandled_intent(),
                    _make_context(),
                    "what is the weather",
                    user_id="default",
                )
            )
    handler.assert_not_called()


def test_free_form_ha_bridge_is_desktop_only_even_with_home_scope() -> None:
    ha = MagicMock()
    ha.enabled = True
    ha.process_transcript.return_value = "Lights changed."
    ha._command_history = None
    dispatcher = _make_dispatcher(ha_bridge=ha)

    for scopes, permissions in (
        (frozenset({"chat.send"}), frozenset()),
        (frozenset({"chat.send", "home.control"}), frozenset({"ha_control"})),
    ):
        with mobile_action_context(scopes, permissions=permissions):
            result = asyncio.run(
                dispatcher.dispatch(
                    _unhandled_intent(),
                    _make_context(),
                    "turn on the lights",
                    user_id="default",
                )
            )
        assert result.response == "LLM reply"

    ha.process_transcript.assert_not_called()


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


def test_free_form_ha_response_post_processing_is_desktop_only() -> None:
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
    assert allowed == "[[ha:light.turn_on entity_id=light.office]]"
    ha.post_process_response.assert_not_called()


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

    with mobile_action_context(
        frozenset({"home.control"}),
        permissions=frozenset({"admin"}),
    ):
        with pytest.raises(MobileStrongAuthRequiredError):
            ToolDispatcher(registry).dispatch(
                "home_assistant_call_service",
                {"domain": "light", "service": "turn_off", "entity_id": "light.office"},
                {"user_id": "default"},
            )
    handler.assert_not_called()


def test_exact_strong_auth_approval_reaches_dispatch_boundary_once() -> None:
    tool, handler = _tool("home_assistant_call_service", ["smart_home"], "mutation")
    registry = ToolRegistry()
    registry.register(tool)
    authority = MagicMock()
    principal = object()
    args = {
        "domain": "light",
        "service": "turn_off",
        "entity_id": "light.office",
    }

    with mobile_action_context(
        frozenset({"home.control"}),
        permissions=frozenset({"admin"}),
        strong_auth_authority=authority,
        strong_auth_principal=principal,
        strong_auth_approval_id="approval-123",
    ):
        result = ToolDispatcher(registry).dispatch(
            "home_assistant_call_service",
            args,
            {"user_id": "default"},
        )

    authority.consume_approval.assert_called_once_with(
        principal,
        approval_id="approval-123",
        action_name="home_assistant_call_service",
        payload=args,
    )
    assert result.status == "attempted_unverified"
    handler.assert_called_once()


def test_one_approval_cannot_execute_same_privileged_action_twice() -> None:
    tool, handler = _tool("home_assistant_call_service", ["smart_home"], "mutation")
    registry = ToolRegistry()
    registry.register(tool)
    authority = MagicMock()
    authority.consume_approval.side_effect = [None, StrongAuthError("approval_replayed", "used")]
    args = {"domain": "light", "service": "turn_off", "entity_id": "light.office"}

    with mobile_action_context(
        frozenset({"home.control"}),
        permissions=frozenset({"admin"}),
        strong_auth_authority=authority,
        strong_auth_principal=object(),
        strong_auth_approval_id="approval-123",
    ):
        first = ToolDispatcher(registry).dispatch(
            "home_assistant_call_service", args, {"user_id": "default"}
        )
        with pytest.raises(MobileStrongAuthRequiredError):
            ToolDispatcher(registry).dispatch(
                "home_assistant_call_service", args, {"user_id": "default"}
            )

    assert first.status == "attempted_unverified"
    assert authority.consume_approval.call_count == 2
    handler.assert_called_once()


def test_concurrent_same_action_cannot_share_one_approval() -> None:
    authority = MagicMock()
    authority.consume_approval.side_effect = [
        None,
        StrongAuthError("approval_replayed", "used"),
    ]
    principal = object()
    args = {"domain": "light", "service": "turn_off", "entity_id": "light.office"}

    async def scenario() -> None:
        first_entered = asyncio.Event()
        release_first = asyncio.Event()

        async def first_execution() -> None:
            with authorized_mobile_tool(
                "home_assistant_call_service",
                operation="mutation",
                arguments=args,
            ):
                first_entered.set()
                await release_first.wait()

        with mobile_action_context(
            frozenset({"home.control"}),
            permissions=frozenset({"admin"}),
            strong_auth_authority=authority,
            strong_auth_principal=principal,
            strong_auth_approval_id="approval-123",
        ):
            first_task = asyncio.create_task(first_execution())
            await first_entered.wait()
            with pytest.raises(MobileStrongAuthRequiredError):
                with authorized_mobile_tool(
                    "home_assistant_call_service",
                    operation="mutation",
                    arguments=args,
                ):
                    pass
            release_first.set()
            await first_task

    asyncio.run(scenario())
    assert authority.consume_approval.call_count == 2


def test_nested_authorization_layers_consume_one_approval_for_one_execution() -> None:
    tool, handler = _tool("home_assistant_call_service", ["smart_home"], "mutation")
    registry = ToolRegistry()
    registry.register(tool)
    authority = MagicMock()
    principal = object()
    args = {"domain": "light", "service": "turn_off", "entity_id": "light.office"}

    with mobile_action_context(
        frozenset({"home.control"}),
        permissions=frozenset({"admin"}),
        strong_auth_authority=authority,
        strong_auth_principal=principal,
        strong_auth_approval_id="approval-123",
    ):
        with authorized_mobile_tool(
            "home_assistant_call_service",
            operation="mutation",
            arguments=args,
        ):
            result = ToolDispatcher(registry).dispatch(
                "home_assistant_call_service", args, {"user_id": "default"}
            )

    authority.consume_approval.assert_called_once()
    assert result.status == "attempted_unverified"
    handler.assert_called_once()


def test_mobile_pre_llm_dispatch_skips_mutations_without_structured_arguments() -> None:
    tool, handler = _tool("home_assistant_call_service", ["smart_home"], "mutation")
    registry = ToolRegistry()
    registry.register(tool)
    dispatcher = _make_dispatcher(tool_dispatcher=ToolDispatcher(registry))

    with mobile_action_context(
        frozenset({"chat.send", "home.control"}),
        permissions=frozenset({"admin"}),
    ):
        result = asyncio.run(
            dispatcher.dispatch(
                _unhandled_intent(),
                _make_context(),
                "turn off the office light",
                user_id="default",
            )
        )

    assert result.response == "LLM reply"
    handler.assert_not_called()
