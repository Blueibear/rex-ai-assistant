from __future__ import annotations

from types import SimpleNamespace

from rex.capabilities.recovery import (
    CapabilityGapResolver,
    ExternalCapabilityCandidate,
    RecoveryActionKind,
)
from rex.capabilities.registry import Capability, CapabilityRegistry


def _cap(
    name: str,
    *,
    enabled: bool = True,
    source: str = "local",
    risk: str = "safe",
    permissions: tuple[str, ...] = (),
    operation: str = "read",
    triggers: list[str] | None = None,
    requires_config: tuple[str, ...] = (),
    requires_identity: bool = False,
    integration_state: str | None = None,
) -> Capability:
    return Capability(
        name=name,
        description=f"Capability for {name.replace('_', ' ')}",
        triggers=triggers or [name.replace("_", " ")],
        enabled=enabled,
        source=source,
        risk=risk,  # type: ignore[arg-type]
        required_permissions=permissions,
        operation=operation,  # type: ignore[arg-type]
        requires_config=requires_config,
        requires_identity=requires_identity,
        integration_state=integration_state,
    )


def test_enabled_local_capability_wins_before_every_external_source() -> None:
    registry = CapabilityRegistry()
    registry.register(_cap("photo_lookup", triggers=["find photo"]))
    registry.register(_cap("openclaw_photo", source="openclaw", triggers=["find photo"]))
    resolver = CapabilityGapResolver(
        registry,
        mcp_candidates=[ExternalCapabilityCandidate("mcp-photo", "mcp", "find photo")],
        openapi_candidates=[ExternalCapabilityCandidate("api-photo", "openapi", "find photo")],
    )

    plan = resolver.resolve("find photo", user_id="james", granted_permissions=set())

    assert plan.searched_sources == ("local_enabled",)
    assert plan.actions[0].kind is RecoveryActionKind.USE_CAPABILITY
    assert plan.actions[0].target == "photo_lookup"


def test_disabled_local_capability_is_offered_before_openclaw() -> None:
    registry = CapabilityRegistry()
    registry.register(_cap("photo_lookup", enabled=False, triggers=["find photo"]))
    registry.register(_cap("openclaw_photo", source="openclaw", triggers=["find photo"]))

    plan = CapabilityGapResolver(registry).resolve(
        "find photo", user_id="james", granted_permissions=set()
    )

    assert plan.searched_sources == ("local_enabled", "local_disabled")
    assert plan.actions[0].kind is RecoveryActionKind.ENABLE_CAPABILITY
    assert plan.actions[0].target == "photo_lookup"


def test_openclaw_then_mcp_then_openapi_order_is_deterministic() -> None:
    registry = CapabilityRegistry()
    registry.register(_cap("openclaw_photo", source="openclaw", triggers=["find photo"]))
    mcp = [ExternalCapabilityCandidate("mcp-photo", "mcp", "find photo")]
    openapi = [ExternalCapabilityCandidate("api-photo", "openapi", "find photo")]

    openclaw_plan = CapabilityGapResolver(
        registry, mcp_candidates=mcp, openapi_candidates=openapi
    ).resolve("find photo", user_id="james", granted_permissions=set())
    assert openclaw_plan.actions[0].source == "openclaw"
    assert openclaw_plan.searched_sources[-1] == "openclaw"

    empty_registry = CapabilityRegistry()
    mcp_plan = CapabilityGapResolver(
        empty_registry, mcp_candidates=mcp, openapi_candidates=openapi
    ).resolve("find photo", user_id="james", granted_permissions=set())
    assert mcp_plan.actions[0].source == "mcp"
    assert mcp_plan.searched_sources[-1] == "mcp"

    openapi_plan = CapabilityGapResolver(empty_registry, openapi_candidates=openapi).resolve(
        "find photo", user_id="james", granted_permissions=set()
    )
    assert openapi_plan.actions[0].source == "openapi"
    assert openapi_plan.searched_sources[-1] == "openapi"


def test_permissioned_candidate_requests_access_instead_of_proposing_execution() -> None:
    registry = CapabilityRegistry()
    registry.register(
        _cap(
            "private_files",
            triggers=["read private files"],
            permissions=("files.private.read",),
            risk="sensitive",
        )
    )

    plan = CapabilityGapResolver(registry).resolve(
        "read private files", user_id="james", granted_permissions=set()
    )

    assert plan.actions[0].kind is RecoveryActionKind.REQUEST_PERMISSION
    assert plan.actions[0].required_permissions == ("files.private.read",)
    assert plan.actions[0].requires_confirmation is True
    assert all(action.kind is not RecoveryActionKind.USE_CAPABILITY for action in plan.actions)


def test_prohibited_candidate_is_not_proposed_or_replaced_with_generated_code() -> None:
    registry = CapabilityRegistry()
    registry.register(_cap("dangerous_action", triggers=["dangerous action"], risk="prohibited"))

    plan = CapabilityGapResolver(registry).resolve(
        "dangerous action", user_id="james", granted_permissions={"admin"}
    )

    assert plan.blocked is True
    assert plan.actions == ()
    assert "prohibited" in plan.message.lower()


def test_safe_declarative_composition_is_after_external_sources() -> None:
    registry = CapabilityRegistry()
    registry.register(_cap("lookup_alpha", triggers=["alpha lookup"]))
    registry.register(_cap("lookup_beta", triggers=["beta lookup"]))

    plan = CapabilityGapResolver(registry).resolve(
        "alpha beta", user_id="james", granted_permissions=set()
    )

    assert plan.actions[0].kind is RecoveryActionKind.COMPOSE_CAPABILITIES
    assert plan.actions[0].targets == ("lookup_alpha", "lookup_beta")
    assert plan.searched_sources == (
        "local_enabled",
        "local_disabled",
        "openclaw",
        "mcp",
        "openapi",
        "composition",
    )


def test_build_is_last_and_requires_confirmation() -> None:
    plan = CapabilityGapResolver(CapabilityRegistry()).resolve(
        "build a new capability to teleport packages", user_id="james", granted_permissions=set()
    )

    assert plan.searched_sources == (
        "local_enabled",
        "local_disabled",
        "openclaw",
        "mcp",
        "openapi",
        "composition",
        "forge",
    )
    assert plan.actions[0].kind is RecoveryActionKind.BUILD_CAPABILITY
    assert plan.actions[0].requires_confirmation is True
    assert "build" in plan.actions[0].label.lower()


def test_recovery_action_serializes_without_hidden_authority() -> None:
    plan = CapabilityGapResolver(CapabilityRegistry()).resolve(
        "build a new capability to teleport packages", user_id="james", granted_permissions=set()
    )

    payload = plan.actions[0].to_dict()
    assert payload["kind"] == "build_capability"
    assert payload["requires_confirmation"] is True
    assert "granted_permissions" not in payload
    assert "user_id" not in payload


def test_disabled_capability_names_specific_missing_config_requirement() -> None:
    registry = CapabilityRegistry()
    registry.register(
        Capability(
            name="weather_lookup",
            description="Read weather",
            triggers=["weather lookup"],
            enabled=False,
            requires_config=("weather_api_key",),
        )
    )

    plan = CapabilityGapResolver(registry).resolve(
        "weather lookup", user_id="james", granted_permissions=set()
    )

    assert plan.actions[0].kind is RecoveryActionKind.ENABLE_CAPABILITY
    assert plan.actions[0].label == "Configure weather_lookup"
    assert "weather_api_key" in plan.actions[0].detail


def test_build_can_be_suppressed_for_non_capability_chitchat() -> None:
    plan = CapabilityGapResolver(CapabilityRegistry()).resolve(
        "what is gravity", user_id="james", granted_permissions=set(), allow_build=False
    )

    assert plan.actions == ()
    assert "forge" not in plan.searched_sources


def test_clawhub_skill_uses_openclaw_registry_source_before_mcp() -> None:
    registry = CapabilityRegistry()
    # ClawHub skills are surfaced by OpenClaw and retain the canonical openclaw source.
    registry.register(_cap("clawhub_photo", source="openclaw", triggers=["find photo"]))
    mcp = [ExternalCapabilityCandidate("mcp-photo", "mcp", "find photo")]

    plan = CapabilityGapResolver(registry, mcp_candidates=mcp).resolve(
        "find photo", user_id="james", granted_permissions=set()
    )

    assert plan.actions[0].source == "openclaw"
    assert plan.actions[0].kind is RecoveryActionKind.CONNECT_PROVIDER
    assert plan.searched_sources == ("local_enabled", "local_disabled", "openclaw")


def test_enabled_capability_with_missing_config_recovers_to_configuration() -> None:
    registry = CapabilityRegistry()
    registry.register(
        _cap(
            "weather_lookup",
            triggers=["check weather"],
            requires_config=("weather_api_key",),
        )
    )

    plan = CapabilityGapResolver(registry, config=SimpleNamespace(weather_api_key=None)).resolve(
        "check weather", user_id="james", granted_permissions=set()
    )

    assert plan.actions[0].kind is RecoveryActionKind.ENABLE_CAPABILITY
    assert plan.actions[0].label == "Configure weather_lookup"
    assert "weather_api_key" in plan.actions[0].detail


def test_argument_bearing_request_still_matches_existing_capability() -> None:
    registry = CapabilityRegistry()
    registry.register(_cap("send_email", enabled=False, triggers=["send email"]))

    plan = CapabilityGapResolver(registry).resolve(
        "send an email to Alice saying hello tomorrow",
        user_id="james",
        granted_permissions=set(),
    )

    assert plan.actions[0].kind is RecoveryActionKind.ENABLE_CAPABILITY
    assert plan.actions[0].target == "send_email"


def test_admin_permission_satisfies_capability_specific_requirements() -> None:
    registry = CapabilityRegistry()
    registry.register(
        _cap(
            "private_files",
            triggers=["read private files"],
            permissions=("files.private.read",),
            risk="sensitive",
        )
    )

    plan = CapabilityGapResolver(registry).resolve(
        "read private files", user_id="james", granted_permissions={"admin"}
    )

    assert plan.actions[0].kind is RecoveryActionKind.USE_CAPABILITY
    assert plan.actions[0].required_permissions == ()


def test_identity_scoped_candidate_requires_identified_user_before_use() -> None:
    registry = CapabilityRegistry()
    registry.register(
        _cap(
            "private_notes",
            triggers=["read private notes"],
            requires_identity=True,
        )
    )

    plan = CapabilityGapResolver(registry).resolve(
        "read private notes", user_id=None, granted_permissions=set()
    )

    assert plan.actions[0].kind is RecoveryActionKind.IDENTIFY_USER
    assert plan.actions[0].target == "private_notes"
    assert all(action.kind is not RecoveryActionKind.USE_CAPABILITY for action in plan.actions)


def test_unhealthy_external_candidate_is_not_offered() -> None:
    resolver = CapabilityGapResolver(
        CapabilityRegistry(),
        mcp_candidates=(
            ExternalCapabilityCandidate(
                "mcp-calendar",
                "mcp",
                "book calendar",
                triggers=("book calendar",),
                health="unavailable",
            ),
        ),
    )

    plan = resolver.resolve(
        "book calendar", user_id="james", granted_permissions=set(), allow_build=False
    )

    assert plan.actions == ()
    assert plan.searched_sources[-1] == "composition"


def test_generic_action_verbs_do_not_match_unrelated_capabilities() -> None:
    registry = CapabilityRegistry()
    registry.register(_cap("send_sms", triggers=["send sms"], permissions=("sms_send",)))
    registry.register(_cap("music_assistant", enabled=False, triggers=["play music"]))

    send_plan = CapabilityGapResolver(registry).resolve(
        "send me an explanation of gravity",
        user_id="james",
        granted_permissions=set(),
        allow_build=False,
    )
    play_plan = CapabilityGapResolver(registry).resolve(
        "play devil's advocate about remote work",
        user_id="james",
        granted_permissions=set(),
        allow_build=False,
    )

    assert send_plan.actions == ()
    assert play_plan.actions == ()


def test_ordinary_creative_request_does_not_offer_forge_build() -> None:
    plan = CapabilityGapResolver(CapabilityRegistry()).resolve(
        "write me a poem about summer",
        user_id="james",
        granted_permissions=set(),
        allow_build=False,
    )

    assert plan.actions == ()
    assert "forge" not in plan.searched_sources


def test_unconfigured_integration_names_missing_integration_requirement() -> None:
    registry = CapabilityRegistry()
    registry.register(
        _cap(
            "calendar_sync",
            triggers=["sync calendar"],
            integration_state="unconfigured",
        )
    )

    plan = CapabilityGapResolver(registry).resolve(
        "sync calendar", user_id="james", granted_permissions=set()
    )

    assert plan.actions[0].kind is RecoveryActionKind.ENABLE_CAPABILITY
    assert plan.actions[0].target == "calendar_sync"
    assert "integration" in plan.message.lower()
    assert "unconfigured" in plan.message.lower()
    assert "configuration" in plan.actions[0].detail.lower()


def test_unavailable_integration_fails_closed_without_fake_setup_path() -> None:
    registry = CapabilityRegistry()
    registry.register(
        _cap(
            "legacy_provider",
            triggers=["legacy provider"],
            integration_state="unavailable",
        )
    )

    plan = CapabilityGapResolver(registry).resolve(
        "legacy provider", user_id="james", granted_permissions=set()
    )

    assert plan.blocked is True
    assert plan.actions == ()
    assert "unavailable" in plan.message.lower()
    assert "no supported configuration path" in plan.message.lower()


def test_disabled_capability_with_satisfied_config_offers_enable_not_reconfigure() -> None:
    registry = CapabilityRegistry()
    registry.register(
        _cap(
            "weather_lookup",
            enabled=False,
            triggers=["check weather"],
            requires_config=("weather_api_key",),
        )
    )

    plan = CapabilityGapResolver(
        registry, config=SimpleNamespace(weather_api_key="configured")
    ).resolve("check weather", user_id="james", granted_permissions=set())

    assert plan.actions[0].kind is RecoveryActionKind.ENABLE_CAPABILITY
    assert plan.actions[0].label == "Enable weather_lookup"
    assert "Enable this capability" in plan.actions[0].detail
    assert "weather_api_key" not in plan.actions[0].detail
