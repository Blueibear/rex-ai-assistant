"""Atomic OpenClaw/ClawHub capability inventory synchronization.

Remote metadata is untrusted input. It may contribute descriptive and schema
metadata for remote-owned cards, but it never grants Rex permissions or
weakens operation/risk/identity/verification authority. The last validated
snapshot is persisted beneath household runtime data for safe stale recovery.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from rex.capabilities.registry import Capability, CapabilityConflictError, CapabilityRegistry
from rex.runtime_paths import household_data_path

if TYPE_CHECKING:
    from rex.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_SNAPSHOT_VERSION = 1
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_MAX_DESCRIPTION = 2048
_MAX_SCHEMA_BYTES = 64 * 1024


class CapabilityInventoryClient(Protocol):
    def fetch_capability_inventory(self, *, session_key: str | None = None) -> dict[str, Any]: ...


class CapabilitySyncValidationError(ValueError):
    """Raised when untrusted remote inventory fails bounded validation."""


@dataclass(frozen=True)
class OpenClawSyncResult:
    success: bool
    stale: bool
    added: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    error_code: str | None = None
    message: str = ""


class OpenClawCapabilitySync:
    """Synchronize one validated remote inventory into the canonical registry."""

    def __init__(
        self,
        registry: CapabilityRegistry,
        inventory_client: CapabilityInventoryClient,
        *,
        snapshot_path: Path | str | None = None,
        session_key: str | None = None,
        tool_registry: ToolRegistry | None = None,
        runtime_config: Any = None,
    ) -> None:
        self._registry = registry
        self._client = inventory_client
        self._tool_registry = tool_registry
        self._runtime_config = runtime_config
        self._snapshot_path = (
            Path(snapshot_path)
            if snapshot_path is not None
            else household_data_path("openclaw", "capability_snapshot.json")
        )
        self._session_key = session_key

    def refresh(self) -> OpenClawSyncResult:
        """Fetch, validate, atomically apply, and persist a fresh remote snapshot."""
        try:
            inventory = self._client.fetch_capability_inventory(session_key=self._session_key)
            normalized = _normalize_inventory(inventory)
            # Durable commit happens before publishing the in-memory registry snapshot.
            # If persistence fails, the previously active safe snapshot is untouched.
            self._persist_snapshot(normalized)
            added, updated, removed = self._apply_snapshot(normalized)
            after = _remote_cards(self._registry)
            logger.info(
                "OpenClaw capability snapshot synchronized: remote=%d added=%d updated=%d removed=%d",
                len(after),
                len(added),
                len(updated),
                len(removed),
                extra={"event": "openclaw.capability_sync", "status": "success"},
            )
            return OpenClawSyncResult(
                success=True,
                stale=False,
                added=added,
                updated=updated,
                removed=removed,
                message="OpenClaw capability snapshot synchronized.",
            )
        except Exception as exc:
            # If this is a fresh process, recover only from our previously validated
            # normalized snapshot. A malformed local snapshot is ignored fail-closed.
            if not _remote_cards(self._registry):
                self.restore_last_safe_snapshot()
            self._mark_openclaw_unavailable()
            error_code = type(exc).__name__
            logger.warning(
                "OpenClaw capability synchronization failed; last safe snapshot marked stale (%s)",
                error_code,
                extra={
                    "event": "openclaw.capability_sync",
                    "status": "stale",
                    "failure": error_code,
                },
            )
            return OpenClawSyncResult(
                success=False,
                stale=True,
                error_code=error_code,
                message="OpenClaw capability synchronization failed; the last safe snapshot is stale.",
            )

    def restore_last_safe_snapshot(self) -> tuple[str, ...]:
        """Restore persisted normalized metadata as unavailable stale evidence."""
        try:
            raw = json.loads(self._snapshot_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("version") != _SNAPSHOT_VERSION:
                raise CapabilitySyncValidationError("unsupported snapshot version")
            records = raw.get("capabilities")
            if not isinstance(records, list):
                raise CapabilitySyncValidationError("snapshot capabilities must be a list")
            capabilities = [_capability_from_snapshot(record) for record in records]
            self._apply_snapshot(capabilities)
            return self._mark_openclaw_unavailable()
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return ()

    def _remote_handler_factory(self, tool_name: str) -> Any:
        config = self._runtime_config
        default_session_key = self._session_key or "agent:main:main"

        def _handler(**kwargs: Any) -> dict[str, Any]:
            from rex.openclaw.errors import OpenClawConfigError
            from rex.openclaw.http_client import get_openclaw_client

            if config is None:
                raise OpenClawConfigError("OpenClaw runtime config is unavailable")
            client = get_openclaw_client(config)
            if client is None:
                raise OpenClawConfigError("OpenClaw gateway is not configured")
            kwargs.pop("context", None)
            kwargs.pop("_user_id", None)
            kwargs.pop("confirmed", None)
            payload = {
                "tool": tool_name,
                "args": kwargs,
                "sessionKey": default_session_key,
            }
            result = client.post("/tools/invoke", json=payload)
            if not isinstance(result, dict):
                raise OpenClawConfigError("OpenClaw tool response was not an object")
            return result

        return _handler

    def _apply_snapshot(
        self, capabilities: list[Capability] | tuple[Capability, ...]
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        if self._tool_registry is not None:
            return self._tool_registry.apply_openclaw_snapshot(
                capabilities, handler_factory=self._remote_handler_factory
            )
        return self._registry.apply_openclaw_snapshot(capabilities)

    def _mark_openclaw_unavailable(self) -> tuple[str, ...]:
        if self._tool_registry is not None:
            return self._tool_registry.mark_openclaw_unavailable(
                handler_factory=self._remote_handler_factory
            )
        return self._registry.mark_openclaw_unavailable()

    def _persist_snapshot(self, capabilities: list[Capability]) -> None:
        payload = {
            "version": _SNAPSHOT_VERSION,
            "capabilities": [_capability_to_snapshot(card) for card in capabilities],
        }
        self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self._snapshot_path.name}.", suffix=".tmp", dir=self._snapshot_path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self._snapshot_path)
        finally:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass


def _remote_cards(registry: CapabilityRegistry) -> dict[str, Capability]:
    return {
        card.id: card for card in registry.list(include_disabled=True) if card.source == "openclaw"
    }


def _normalize_inventory(inventory: dict[str, Any]) -> list[Capability]:
    if not isinstance(inventory, dict):
        raise CapabilitySyncValidationError("inventory must be an object")
    tools_catalog = inventory.get("tools_catalog")
    skills_status = inventory.get("skills_status")
    effective_tools = inventory.get("effective_tools")
    if not isinstance(tools_catalog, dict):
        raise CapabilitySyncValidationError("tools.catalog payload must be an object")
    if not isinstance(skills_status, dict):
        raise CapabilitySyncValidationError("skills.status payload must be an object")

    effective = _effective_tool_ids(effective_tools)
    capabilities: list[Capability] = []
    seen: set[str] = set()
    for raw_tool in _iter_catalog_tools(tools_catalog):
        name = _validated_id(raw_tool.get("id", raw_tool.get("name")))
        if name in seen:
            raise CapabilityConflictError(f"Duplicate OpenClaw capability in inventory: {name!r}")
        seen.add(name)
        description = _validated_description(raw_tool.get("description"), f"OpenClaw tool {name}")
        schema = _flatten_schema(raw_tool.get("inputSchema", raw_tool.get("input_schema", {})))
        is_effective = effective is not None and name in effective
        capabilities.append(
            _remote_card(
                name=name,
                description=description,
                category="OpenClaw Tool",
                input_schema=schema,
                enabled=is_effective,
                health=("healthy" if is_effective else "unavailable"),
                integration_state=None if is_effective else "unavailable",
            )
        )

    skills = skills_status.get("skills", [])
    if not isinstance(skills, list):
        raise CapabilitySyncValidationError("skills.status skills must be a list")
    for raw_skill in skills:
        if not isinstance(raw_skill, dict):
            raise CapabilitySyncValidationError("skill metadata must be an object")
        skill_name = _validated_id(raw_skill.get("name", raw_skill.get("id")))
        capability_id = f"openclaw_skill__{skill_name}"
        if capability_id in seen:
            raise CapabilityConflictError(
                f"Duplicate OpenClaw capability in inventory: {capability_id!r}"
            )
        seen.add(capability_id)
        description = _validated_description(
            raw_skill.get("description"), f"OpenClaw skill {skill_name}"
        )
        eligible = raw_skill.get("eligible") is True
        capabilities.append(
            _remote_card(
                name=capability_id,
                description=description,
                category="OpenClaw Skill",
                input_schema={},
                enabled=False,
                health="unavailable",
                integration_state="informational" if eligible else "unavailable",
                triggers=(skill_name.replace("-", " "),),
            )
        )
    return capabilities


def _iter_catalog_tools(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    tools = catalog.get("tools")
    if tools is not None:
        if not isinstance(tools, list) or not all(isinstance(item, dict) for item in tools):
            raise CapabilitySyncValidationError("tools.catalog tools must be a list of objects")
        return list(tools)
    groups = catalog.get("groups", [])
    if not isinstance(groups, list):
        raise CapabilitySyncValidationError("tools.catalog groups must be a list")
    result: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            raise CapabilitySyncValidationError("tool group must be an object")
        group_tools = group.get("tools", [])
        if not isinstance(group_tools, list) or not all(
            isinstance(item, dict) for item in group_tools
        ):
            raise CapabilitySyncValidationError("tool group tools must be a list of objects")
        result.extend(group_tools)
    return result


def _effective_tool_ids(payload: Any) -> set[str] | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise CapabilitySyncValidationError("tools.effective payload must be an object")
    result: set[str] = set()
    if "found" in payload:
        found = payload.get("found")
        if not isinstance(found, list):
            raise CapabilitySyncValidationError("tools.effective found must be a list")
        for group in found:
            if (
                not isinstance(group, list)
                or len(group) != 2
                or not isinstance(group[0], str)
                or not isinstance(group[1], list)
            ):
                raise CapabilitySyncValidationError("tools.effective found group is invalid")
            for tool_id in group[1]:
                result.add(_validated_id(tool_id))
        return result
    raw_tools = payload.get("tools")
    if raw_tools is None:
        raise CapabilitySyncValidationError("tools.effective missing found inventory")
    if not isinstance(raw_tools, list):
        raise CapabilitySyncValidationError("tools.effective tools must be a list")
    for item in raw_tools:
        value = item.get("id", item.get("name")) if isinstance(item, dict) else item
        result.add(_validated_id(value))
    return result


def _validated_id(value: Any) -> str:
    if not isinstance(value, str):
        raise CapabilitySyncValidationError("capability ID must be a string")
    name = value.strip()
    if not _ID_RE.fullmatch(name):
        raise CapabilitySyncValidationError("capability ID is missing or invalid")
    return name


def _validated_description(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    if not isinstance(value, str):
        raise CapabilitySyncValidationError("capability description must be a string")
    description = value.strip() or fallback
    if len(description) > _MAX_DESCRIPTION or any(
        ord(char) < 32 and char not in "\t\n\r" for char in description
    ):
        raise CapabilitySyncValidationError("capability description is invalid or too large")
    return description


def _flatten_schema(value: Any) -> dict[str, str]:
    if value in (None, {}):
        return {}
    if not isinstance(value, dict):
        raise CapabilitySyncValidationError("capability input schema must be an object")
    try:
        encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise CapabilitySyncValidationError("capability input schema is not JSON-safe") from exc
    if len(encoded.encode("utf-8")) > _MAX_SCHEMA_BYTES:
        raise CapabilitySyncValidationError("capability input schema is too large")

    # Rex's canonical Tool Card schema is a bounded name->type mapping. Normalize
    # JSON Schema objects without retaining remote descriptions/defaults/examples.
    properties = value.get("properties")
    if properties is None:
        if all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
            return dict(sorted(value.items()))
        return {}
    if not isinstance(properties, dict):
        raise CapabilitySyncValidationError("capability schema properties must be an object")
    result: dict[str, str] = {}
    for raw_name, raw_spec in properties.items():
        name = _validated_schema_key(raw_name)
        if not isinstance(raw_spec, dict):
            raise CapabilitySyncValidationError("capability schema property must be an object")
        raw_type = raw_spec.get("type", "any")
        if isinstance(raw_type, list):
            types = [item for item in raw_type if isinstance(item, str)]
            type_name = "|".join(types[:4]) if types else "any"
        elif isinstance(raw_type, str):
            type_name = raw_type
        else:
            type_name = "any"
        result[name] = type_name[:80] or "any"
    return dict(sorted(result.items()))


def _validated_schema_key(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise CapabilitySyncValidationError("capability schema key is invalid")
    if any(ord(char) < 32 for char in value):
        raise CapabilitySyncValidationError("capability schema key is invalid")
    return value


def _remote_card(
    *,
    name: str,
    description: str,
    category: str,
    input_schema: dict[str, str],
    enabled: bool,
    health: str,
    integration_state: str | None,
    triggers: tuple[str, ...] = (),
) -> Capability:
    inferred_trigger = name.replace("_", " ").replace(".", " ").replace(":", " ")
    return Capability(
        name=name,
        description=description,
        triggers=list(dict.fromkeys((*triggers, inferred_trigger))),
        enabled=enabled,
        category=category,
        integration_state=integration_state,
        read_capable=False,
        write_capable=False,
        source="openclaw",
        input_schema=input_schema,
        output_schema={},
        required_permissions=("openclaw_execute",),
        health=health,
        operation="mutation",
        risk="sensitive",
        verification_supported=False,
        requires_identity=True,
    )


def _capability_to_snapshot(card: Capability) -> dict[str, Any]:
    return {
        "id": card.id,
        "description": card.description,
        "category": card.category,
        "triggers": list(card.triggers),
        "input_schema": dict(card.input_schema),
        "enabled": card.enabled,
        "health": card.health,
        "integration_state": card.integration_state,
    }


def _capability_from_snapshot(value: Any) -> Capability:
    if not isinstance(value, dict):
        raise CapabilitySyncValidationError("snapshot capability must be an object")
    name = _validated_id(value.get("id"))
    description = _validated_description(value.get("description"), f"OpenClaw capability {name}")
    category = value.get("category", "OpenClaw Tool")
    if not isinstance(category, str) or len(category) > 128:
        raise CapabilitySyncValidationError("snapshot category is invalid")
    raw_triggers = value.get("triggers", [])
    if not isinstance(raw_triggers, list) or not all(
        isinstance(item, str) for item in raw_triggers
    ):
        raise CapabilitySyncValidationError("snapshot triggers are invalid")
    schema = _flatten_schema(value.get("input_schema", {}))
    enabled = value.get("enabled") is True
    health = value.get("health")
    if health not in {"unknown", "healthy", "degraded", "unhealthy", "unavailable"}:
        raise CapabilitySyncValidationError("snapshot health is invalid")
    integration_state = value.get("integration_state")
    if integration_state is not None and not isinstance(integration_state, str):
        raise CapabilitySyncValidationError("snapshot integration state is invalid")
    return _remote_card(
        name=name,
        description=description,
        category=category,
        input_schema=schema,
        enabled=enabled,
        health=health,
        integration_state=integration_state,
        triggers=tuple(raw_triggers),
    )


OPENCLAW_CAPABILITY_REFRESH_EVENT = "openclaw.capabilities.refresh_requested"
OPENCLAW_CAPABILITY_REFRESHED_EVENT = "openclaw.capabilities.refreshed"


class OpenClawCapabilitySyncController:
    """Own startup/manual/event refresh for one canonical registry snapshot."""

    def __init__(
        self,
        sync: OpenClawCapabilitySync,
        *,
        event_bus: Any = None,
    ) -> None:
        self._sync = sync
        self._event_bus = event_bus
        self._lock = threading.RLock()
        self._subscribed = False
        self._closed = False
        self._last_result: OpenClawSyncResult | None = None
        self._event_handler = lambda event: self.refresh(reason="hot_refresh")

    @property
    def last_result(self) -> OpenClawSyncResult | None:
        return self._last_result

    def start(self) -> OpenClawSyncResult:
        """Subscribe the hot-refresh seam and perform the startup sync once."""
        with self._lock:
            if self._closed:
                return self._closed_result()
            if self._event_bus is not None and not self._subscribed:
                self._event_bus.subscribe(OPENCLAW_CAPABILITY_REFRESH_EVENT, self._event_handler)
                self._subscribed = True
            return self.refresh(reason="startup")

    def _closed_result(self) -> OpenClawSyncResult:
        return OpenClawSyncResult(
            success=False,
            stale=True,
            error_code="ControllerClosed",
            message="OpenClaw capability refresh ignored because this controller is no longer current.",
        )

    def refresh(self, *, reason: str = "manual") -> OpenClawSyncResult:
        """Run one synchronous refresh and publish privacy-safe result metadata."""
        with self._lock:
            if self._closed:
                return self._closed_result()
            result = self._sync.refresh()
            self._last_result = result
        logger.info(
            "OpenClaw capability refresh finished: reason=%s success=%s stale=%s",
            reason,
            result.success,
            result.stale,
            extra={
                "event": "openclaw.capability_refresh",
                "reason": reason,
                "success": result.success,
                "stale": result.stale,
            },
        )
        if self._event_bus is not None:
            self._event_bus.publish(
                OPENCLAW_CAPABILITY_REFRESHED_EVENT,
                {
                    "reason": reason,
                    "success": result.success,
                    "stale": result.stale,
                    "added": len(result.added),
                    "updated": len(result.updated),
                    "removed": len(result.removed),
                    "error_code": result.error_code,
                },
            )
        return result

    def close(self) -> None:
        """Remove the hot-refresh subscription without touching registry state."""
        with self._lock:
            self._closed = True
            if self._event_bus is not None and self._subscribed:
                self._event_bus.unsubscribe(OPENCLAW_CAPABILITY_REFRESH_EVENT, self._event_handler)
            self._subscribed = False


class _FailingInventoryClient:
    """Adapter that routes setup failures through normal stale-snapshot recovery."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def fetch_capability_inventory(self, *, session_key: str | None = None) -> dict[str, Any]:
        raise self._error


def _disabled_remote_handler(tool_name: str) -> Any:
    def _handler(**kwargs: Any) -> dict[str, Any]:
        from rex.openclaw.errors import OpenClawConfigError

        raise OpenClawConfigError(f"OpenClaw capability {tool_name!r} is disabled")

    return _handler


_CONTROLLER_LOCK = threading.RLock()
_CONTROLLER: OpenClawCapabilitySyncController | None = None
_CONTROLLER_KEY: tuple[object, ...] | None = None


def _openclaw_config(config: Any) -> tuple[bool, str, str, float]:
    enabled = bool(getattr(config, "use_openclaw_tools", False))
    integrations = getattr(config, "integrations", None)
    gateway_url = str(
        getattr(integrations, "openclaw_gateway_url", "")
        if integrations is not None
        else getattr(config, "openclaw_gateway_url", "")
    ).strip()
    token = str(getattr(config, "openclaw_gateway_token", "") or "").strip()
    timeout_value = (
        getattr(integrations, "openclaw_gateway_timeout", None)
        if integrations is not None
        else None
    )
    if timeout_value is None:
        timeout_value = getattr(config, "openclaw_gateway_timeout", 5.0)
    if isinstance(timeout_value, (int, float, str)):
        try:
            timeout = max(0.1, float(timeout_value))
        except ValueError:
            timeout = 5.0
    else:
        timeout = 5.0
    return enabled, gateway_url, token, timeout


def initialize_openclaw_capability_sync(
    config: Any,
    *,
    registry: CapabilityRegistry | None = None,
    tool_registry: ToolRegistry | None = None,
    inventory_client: CapabilityInventoryClient | None = None,
    event_bus: Any = None,
    snapshot_path: Path | str | None = None,
    session_key: str | None = None,
) -> OpenClawSyncResult | None:
    """Initialize one process-wide OpenClaw capability lifecycle.

    Disabled OpenClaw never performs discovery and any previously learned remote
    cards are made unavailable. Enabled setup failures flow through the same
    stale-snapshot path as runtime failures so core local Rex can still start.
    """
    from rex.capabilities.registry import get_capability_registry

    if tool_registry is not None:
        resolved_registry = tool_registry.capability_registry
        if registry is not None and registry is not resolved_registry:
            raise ValueError("tool_registry and registry must share the same CapabilityRegistry")
    else:
        resolved_registry = registry if registry is not None else get_capability_registry(config)
    enabled, gateway_url, token, timeout = _openclaw_config(config)

    global _CONTROLLER, _CONTROLLER_KEY
    if not enabled:
        with _CONTROLLER_LOCK:
            if _CONTROLLER is not None:
                _CONTROLLER.close()
            _CONTROLLER = None
            _CONTROLLER_KEY = None
            if tool_registry is not None:
                tool_registry.apply_openclaw_snapshot(
                    (), handler_factory=lambda name: _disabled_remote_handler(name)
                )
            else:
                resolved_registry.apply_openclaw_snapshot(())
            return None

    if event_bus is None:
        from rex.openclaw.event_bus import get_event_bus

        event_bus = get_event_bus()

    resolved_session_key = session_key or "agent:main:main"
    injected_client_id = id(inventory_client) if inventory_client is not None else None
    key = (
        id(resolved_registry),
        id(tool_registry) if tool_registry is not None else None,
        gateway_url,
        hashlib.sha256(token.encode("utf-8")).hexdigest(),
        timeout,
        resolved_session_key,
        str(Path(snapshot_path)) if snapshot_path is not None else None,
        injected_client_id,
        id(event_bus),
    )
    with _CONTROLLER_LOCK:
        if _CONTROLLER is not None and _CONTROLLER_KEY == key:
            return _CONTROLLER.last_result
        if _CONTROLLER is not None:
            _CONTROLLER.close()

        client = inventory_client
        if client is None:
            try:
                from rex.openclaw.gateway_rpc import OpenClawGatewayRpcClient

                client = OpenClawGatewayRpcClient(
                    gateway_url,
                    token,
                    timeout=timeout,
                )
            except Exception as exc:
                client = _FailingInventoryClient(exc)
        sync = OpenClawCapabilitySync(
            resolved_registry,
            client,
            snapshot_path=snapshot_path,
            session_key=resolved_session_key,
            tool_registry=tool_registry,
            runtime_config=config,
        )
        controller = OpenClawCapabilitySyncController(sync, event_bus=event_bus)
        result = controller.start()
        _CONTROLLER = controller
        _CONTROLLER_KEY = key
        return result


def refresh_openclaw_capabilities() -> OpenClawSyncResult | None:
    """Manually refresh the configured process-wide OpenClaw capability snapshot."""
    with _CONTROLLER_LOCK:
        controller = _CONTROLLER
    if controller is None:
        return None
    return controller.refresh(reason="manual")


def reset_openclaw_capability_sync() -> None:
    """Reset process-wide lifecycle state for tests and controlled reconfiguration."""
    global _CONTROLLER, _CONTROLLER_KEY
    with _CONTROLLER_LOCK:
        if _CONTROLLER is not None:
            _CONTROLLER.close()
        _CONTROLLER = None
        _CONTROLLER_KEY = None


__all__ = [
    "CapabilitySyncValidationError",
    "OPENCLAW_CAPABILITY_REFRESH_EVENT",
    "OPENCLAW_CAPABILITY_REFRESHED_EVENT",
    "OpenClawCapabilitySync",
    "OpenClawCapabilitySyncController",
    "OpenClawSyncResult",
    "initialize_openclaw_capability_sync",
    "refresh_openclaw_capabilities",
    "reset_openclaw_capability_sync",
]
