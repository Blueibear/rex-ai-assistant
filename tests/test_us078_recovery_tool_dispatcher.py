from __future__ import annotations

from rex.capabilities.recovery import ExternalCapabilityCandidate, RecoveryActionKind
from rex.tools.dispatcher import ToolDispatcher
from rex.tools.registry import Tool, ToolRegistry


def _tool(
    name: str,
    *,
    enabled: bool = True,
    requires_config: tuple[str, ...] = (),
    permissions: tuple[str, ...] = (),
    risk: str = "safe",
    triggers: list[str] | None = None,
) -> Tool:
    return Tool(
        name=name,
        description=name.replace("_", " "),
        capability_tags=triggers or [name.replace("_", " ")],
        handler=lambda **kwargs: "ok",
        enabled=enabled,
        requires_config=list(requires_config),
        required_permissions=permissions,
        risk=risk,  # type: ignore[arg-type]
    )


def test_disabled_tool_recovery_names_missing_config() -> None:
    registry = ToolRegistry()
    registry.register(_tool("weather_lookup", enabled=False, requires_config=("weather_api_key",)))
    dispatcher = ToolDispatcher(registry, config=object())

    assert dispatcher.select_tools("check weather", user_id="james") == []
    plan = dispatcher.recovery_plan("check weather", user_id="james", granted_permissions=set())

    assert plan is not None
    assert plan.actions[0].kind is RecoveryActionKind.ENABLE_CAPABILITY
    assert "weather_api_key" in plan.actions[0].detail


def test_permissioned_tool_recovery_requests_access() -> None:
    registry = ToolRegistry()
    registry.register(_tool("private_files", permissions=("files.private.read",), risk="sensitive"))
    dispatcher = ToolDispatcher(registry)

    assert (
        dispatcher.select_tools("read private files", user_id="james", granted_permissions=set())
        == []
    )
    plan = dispatcher.recovery_plan(
        "read private files", user_id="james", granted_permissions=set()
    )

    assert plan is not None
    assert plan.actions[0].kind is RecoveryActionKind.REQUEST_PERMISSION
    assert plan.actions[0].required_permissions == ("files.private.read",)


def test_general_knowledge_question_does_not_offer_capability_build() -> None:
    plan = ToolDispatcher(ToolRegistry()).recovery_plan(
        "what is gravity", user_id="james", granted_permissions=set()
    )

    assert plan is None


def test_action_request_without_candidate_offers_confirmation_gated_build() -> None:
    plan = ToolDispatcher(ToolRegistry()).recovery_plan(
        "build a new capability to teleport packages", user_id="james", granted_permissions=set()
    )

    assert plan is not None
    assert plan.actions[0].kind is RecoveryActionKind.BUILD_CAPABILITY
    assert plan.actions[0].requires_confirmation is True


def test_dispatcher_accepts_configured_mcp_candidate_after_local_sources() -> None:
    dispatcher = ToolDispatcher(
        ToolRegistry(),
        mcp_candidates=(
            ExternalCapabilityCandidate(
                "mcp-calendar",
                "mcp",
                "calendar booking",
                triggers=("book calendar",),
                settings_route="/settings?section=integrations",
            ),
        ),
    )

    plan = dispatcher.recovery_plan("book calendar", user_id="james", granted_permissions=set())

    assert plan is not None
    assert plan.actions[0].kind is RecoveryActionKind.CONNECT_PROVIDER
    assert plan.actions[0].source == "mcp"
    assert plan.actions[0].settings_route == "/settings?section=integrations"
    assert plan.searched_sources == (
        "local_enabled",
        "local_disabled",
        "openclaw",
        "mcp",
    )


def test_natural_missing_web_search_request_points_to_existing_config() -> None:
    from rex.tools.registry import get_default_registry

    config = type("Config", (), {"search_providers": "", "tool_timeout_seconds": 10})()
    dispatcher = ToolDispatcher(get_default_registry(), config=config)

    assert dispatcher.select_tools("search web for cats", user_id="james") == []
    plan = dispatcher.recovery_plan(
        "search web for cats", user_id="james", granted_permissions=set()
    )

    assert plan is not None
    assert plan.actions[0].kind is RecoveryActionKind.ENABLE_CAPABILITY
    assert plan.actions[0].target == "web_search"
    assert "search_providers" in plan.actions[0].detail


def test_ordinary_creative_action_word_does_not_offer_build() -> None:
    plan = ToolDispatcher(ToolRegistry()).recovery_plan(
        "write me a poem about summer", user_id="james", granted_permissions=set()
    )

    assert plan is None


def test_informational_tool_topic_does_not_trigger_permission_recovery() -> None:
    registry = ToolRegistry()
    registry.register(
        _tool(
            "send_email",
            permissions=("email_send",),
            risk="sensitive",
            triggers=["send email"],
        )
    )
    dispatcher = ToolDispatcher(registry)

    plan = dispatcher.recovery_plan(
        "how does email work?", user_id="james", granted_permissions=set()
    )

    assert plan is None


def test_creative_email_draft_does_not_trigger_send_permission_recovery() -> None:
    registry = ToolRegistry()
    registry.register(
        _tool(
            "send_email",
            permissions=("email_send",),
            risk="sensitive",
            triggers=["send email"],
        )
    )
    dispatcher = ToolDispatcher(registry)

    plan = dispatcher.recovery_plan(
        "write me an email to my boss", user_id="james", granted_permissions=set()
    )

    assert plan is None


def test_conversational_action_verbs_do_not_trigger_recovery() -> None:
    registry = ToolRegistry()
    registry.register(
        _tool(
            "send_email",
            permissions=("email_send",),
            risk="sensitive",
            triggers=["send email"],
        )
    )
    dispatcher = ToolDispatcher(registry)

    for request in (
        "find an explanation of email security",
        "open a discussion about email security",
        "connect the concept of email to privacy",
        "open the email settings page",
    ):
        assert dispatcher.recovery_plan(request, user_id="james", granted_permissions=set()) is None
