from __future__ import annotations

import pytest

from rex.capabilities.registry import (
    Capability,
    CapabilityConflictError,
    CapabilityRegistry,
    SecurityClassificationError,
)
from rex.tools.registry import Tool, ToolRegistry


def _handler(**_kwargs):
    return {"ok": True}


def _card(**overrides) -> Capability:
    values = {
        "name": "web_lookup",
        "description": "Search trusted web sources.",
        "source": "local",
        "input_schema": {"query": "string"},
        "output_schema": {"results": "array"},
        "required_permissions": ("web.read",),
        "health": "healthy",
        "operation": "read",
        "risk": "safe",
        "verification_supported": True,
        "examples": ("look up the latest release",),
        "triggers": ["search", "look up"],
    }
    values.update(overrides)
    return Capability(**values)


def test_canonical_card_records_required_metadata_and_legacy_aliases() -> None:
    card = _card()

    assert card.id == "web_lookup"
    assert card.source == "local"
    assert card.input_schema == {"query": "string"}
    assert card.output_schema == {"results": "array"}
    assert card.inputs == ["query"]
    assert card.outputs == ["results"]
    assert card.required_permissions == ("web.read",)
    assert card.health == "healthy"
    assert card.operation == "read"
    assert card.risk == "safe"
    assert card.verification_supported is True
    assert card.examples == ("look up the latest release",)


def test_static_security_metadata_is_sealed_but_runtime_state_can_change() -> None:
    card = _card(enabled=True, health="unknown")

    card.enabled = False
    card.health = "degraded"
    assert card.enabled is False
    assert card.health == "degraded"

    with pytest.raises(AttributeError, match="sealed"):
        card.risk = "sensitive"
    with pytest.raises(AttributeError, match="sealed"):
        card.required_permissions = ()


def test_duplicate_id_is_idempotent_only_when_metadata_matches() -> None:
    registry = CapabilityRegistry()
    first = _card()
    registry.register(first)
    registry.register(_card())

    assert registry.get("web_lookup") is first
    with pytest.raises(CapabilityConflictError, match="web_lookup"):
        registry.register(_card(description="Different metadata"))


def test_registry_snapshot_is_deterministic() -> None:
    registry = CapabilityRegistry()
    registry.register(_card(name="zeta", description="Zeta"))
    registry.register(_card(name="alpha", description="Alpha"))

    first = registry.metadata_snapshot()
    second = registry.metadata_snapshot()

    assert first == second
    assert [item["id"] for item in first] == ["alpha", "zeta"]


def test_authorization_is_evaluated_from_current_permissions_each_time() -> None:
    registry = CapabilityRegistry()
    registry.register(_card(required_permissions=("web.read", "network.use")))

    assert registry.is_authorized("web_lookup", {"web.read", "network.use"}) is True
    assert registry.is_authorized("web_lookup", {"web.read"}) is False
    assert registry.is_authorized("web_lookup", {"web.read", "network.use"}) is True


def test_tool_registry_binds_handler_to_canonical_card() -> None:
    cards = CapabilityRegistry()
    registry = ToolRegistry(capability_registry=cards)
    tool = Tool(
        name="system_read",
        description="Read system information.",
        capability_tags=["system", "read"],
        requires_config=[],
        handler=_handler,
        input_schema={"detail": "string"},
        output_schema={"system": "object"},
        required_permissions=("system.read",),
        health="healthy",
        examples=("show system information",),
    )

    registry.register(tool)
    card = cards.get("system_read")

    assert registry.get("system_read") is tool
    assert card is not None
    assert card.input_schema == {"detail": "string"}
    assert card.required_permissions == ("system.read",)
    assert card.operation == "read"
    assert card.risk == "safe"


def test_tool_duplicate_schema_drift_does_not_silently_replace() -> None:
    registry = ToolRegistry()
    registry.register(Tool("dup", "First", [], [], _handler))

    with pytest.raises(CapabilityConflictError, match="dup"):
        registry.register(Tool("dup", "Second", [], [], _handler))


def test_remote_metadata_cannot_weaken_local_security_classification() -> None:
    cards = CapabilityRegistry()
    registry = ToolRegistry(capability_registry=cards)
    registry.register(
        Tool(
            name="set_volume",
            description="Set volume.",
            capability_tags=["audio"],
            requires_config=[],
            handler=_handler,
            operation="mutation",
            risk="sensitive",
            requires_identity=True,
            required_permissions=("system.write",),
        )
    )
    remote = _card(
        name="set_volume",
        description="Remote claims this is harmless.",
        source="openclaw",
        operation="read",
        risk="safe",
        required_permissions=(),
        verification_supported=False,
    )

    with pytest.raises(SecurityClassificationError, match="set_volume"):
        registry.register_remote_card(remote)

    card = cards.get("set_volume")
    assert card is not None
    assert card.source == "local"
    assert card.operation == "mutation"
    assert card.risk == "sensitive"
    assert card.required_permissions == ("system.write",)


def test_remote_matching_security_may_not_replace_local_authority() -> None:
    cards = CapabilityRegistry()
    registry = ToolRegistry(capability_registry=cards)
    registry.register(
        Tool(
            name="local_read",
            description="Local description.",
            capability_tags=["read"],
            requires_config=[],
            handler=_handler,
            required_permissions=("data.read",),
        )
    )
    remote = _card(
        name="local_read",
        description="Remote description.",
        source="openclaw",
        required_permissions=("data.read",),
        verification_supported=False,
    )

    result = registry.register_remote_card(remote)

    assert result.description == "Local description."
    assert cards.get("local_read") is result
    assert result.source == "local"


def test_selection_filters_by_current_permission_snapshot() -> None:
    from rex.tools.dispatcher import ToolDispatcher

    cards = CapabilityRegistry()
    tools = ToolRegistry(capability_registry=cards)
    tools.register(
        Tool(
            name="send_email",
            description="Send email.",
            capability_tags=["email"],
            requires_config=[],
            handler=lambda **_kwargs: {"ok": True},
            required_permissions=("email_send",),
        )
    )
    dispatcher = ToolDispatcher(tools)
    assert dispatcher.select_tools("send email", granted_permissions=frozenset()) == []
    selected = dispatcher.select_tools("send email", granted_permissions=frozenset({"email_send"}))
    assert [tool.name for tool in selected] == ["send_email"]


def test_execution_rechecks_permission_snapshot() -> None:
    from rex.tools.dispatcher import ToolDispatcher
    from rex.tools.execution import ToolOutcome

    calls: list[str] = []
    tools = ToolRegistry()
    tools.register(
        Tool(
            name="permissioned_read",
            description="Permissioned read.",
            capability_tags=["search"],
            requires_config=[],
            handler=lambda **_kwargs: calls.append("called") or {"ok": True},
            required_permissions=("computer_control",),
        )
    )
    dispatcher = ToolDispatcher(tools)
    denied = dispatcher.dispatch(
        "permissioned_read",
        {},
        {"request_id": "denied", "granted_permissions": frozenset()},
    )
    assert denied.status == ToolOutcome.DENIED
    assert calls == []
    allowed = dispatcher.dispatch(
        "permissioned_read",
        {},
        {"request_id": "allowed", "granted_permissions": frozenset({"admin"})},
    )
    assert allowed.status == ToolOutcome.COMPLETED
    assert calls == ["called"]


def test_default_tool_registry_uses_global_capability_authority() -> None:
    from rex.capabilities.registry import (
        get_capability_registry,
        reset_capability_registry,
    )
    from rex.tools.registry import get_default_registry

    reset_capability_registry()
    cards = get_capability_registry()
    tools = get_default_registry()
    assert tools.capability_registry is cards
    tool_names = {tool.name for tool in tools.all_tools()}
    card_names = {card.name for card in cards.list(include_disabled=True)}
    assert tool_names.issubset(card_names)
    assert "chat" in card_names


def test_capability_reset_rebinds_default_tool_registry() -> None:
    from rex.capabilities.registry import (
        get_capability_registry,
        reset_capability_registry,
    )
    from rex.tools.registry import get_default_registry

    reset_capability_registry()
    first_cards = get_capability_registry()
    first_tools = get_default_registry()
    reset_capability_registry()
    second_cards = get_capability_registry()
    second_tools = get_default_registry()
    assert second_cards is not first_cards
    assert second_tools is not first_tools
    assert second_tools.capability_registry is second_cards


def test_openclaw_adapter_cannot_replace_local_security_metadata() -> None:
    from rex.capabilities.registry import get_capability_registry, reset_capability_registry
    from rex.openclaw.tool_registry import ToolMeta as OpenClawToolMeta
    from rex.openclaw.tool_registry import ToolRegistry as OpenClawRegistry
    from rex.tools.registry import get_default_registry

    reset_capability_registry()
    cards = get_capability_registry()
    before = cards.get("send_email")
    assert before is not None and before.operation == "mutation"
    OpenClawRegistry(canonical_registry=get_default_registry()).register_tool(
        OpenClawToolMeta(name="send_email", description="Remote says email is harmless")
    )
    after = cards.get("send_email")
    assert after is before
    assert after.source == "local"
    assert after.operation == "mutation"
    assert after.requires_identity is True


def test_unknown_openclaw_tool_defaults_to_conservative_security() -> None:
    from rex.capabilities.registry import get_capability_registry, reset_capability_registry
    from rex.openclaw.tool_registry import ToolMeta as OpenClawToolMeta
    from rex.openclaw.tool_registry import ToolRegistry as OpenClawRegistry
    from rex.tools.registry import get_default_registry

    reset_capability_registry()
    cards = get_capability_registry()
    OpenClawRegistry(canonical_registry=get_default_registry()).register_tool(
        OpenClawToolMeta(name="remote_unknown", description="Unknown remote tool")
    )
    card = cards.get("remote_unknown")
    assert card is not None
    assert card.source == "openclaw"
    assert card.operation == "mutation"
    assert card.risk == "sensitive"
    assert card.requires_identity is True
    assert card.verification_supported is False


def test_builtin_tool_cards_use_canonical_permission_vocabulary() -> None:
    from rex.capabilities.registry import get_capability_registry, reset_capability_registry

    reset_capability_registry()
    cards = get_capability_registry()
    expected = {
        "send_email": ("email_send",),
        "send_sms": ("sms_send",),
        "home_assistant_call_service": ("ha_control",),
        "music_play": ("ha_control",),
        "file_ops": ("computer_control",),
        "get_system_info": ("computer_control",),
        "set_volume": ("computer_control",),
        "run_sfc_scan": ("computer_control",),
    }
    for name, permissions in expected.items():
        card = cards.get(name)
        assert card is not None
        assert card.required_permissions == permissions


def test_openclaw_health_evidence_updates_canonical_runtime_state() -> None:
    from rex.capabilities.registry import CapabilityRegistry
    from rex.openclaw.tool_registry import ToolMeta as OpenClawToolMeta
    from rex.openclaw.tool_registry import ToolRegistry as OpenClawRegistry
    from rex.tools.registry import ToolRegistry as CanonicalToolRegistry

    cards = CapabilityRegistry()
    canonical = CanonicalToolRegistry(capability_registry=cards)
    remote = OpenClawRegistry(canonical_registry=canonical)
    remote.register_tool(
        OpenClawToolMeta(
            name="remote_health",
            description="Remote health test.",
            health_check=lambda: (False, "gateway unavailable"),
        )
    )
    ok, detail = remote.check_health("remote_health")
    assert ok is False
    assert detail == "gateway unavailable"
    card = cards.get("remote_health")
    assert card is not None
    assert card.health == "unhealthy"


def test_promoted_openclaw_registry_rebinds_to_global_authority() -> None:
    from rex.capabilities.registry import get_capability_registry, reset_capability_registry
    from rex.openclaw.tool_registry import (
        ToolMeta as OpenClawToolMeta,
    )
    from rex.openclaw.tool_registry import (
        ToolRegistry as OpenClawRegistry,
    )
    from rex.openclaw.tool_registry import (
        set_tool_registry,
    )

    reset_capability_registry()
    custom = OpenClawRegistry()
    custom.register_tool(OpenClawToolMeta(name="promoted_remote", description="Promoted"))
    set_tool_registry(custom)
    card = get_capability_registry().get("promoted_remote")
    assert card is not None
    assert card.source == "openclaw"
