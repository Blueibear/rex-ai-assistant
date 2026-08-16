"""Capability registry for AskRex assistant.

Provides a structured registry of all Rex capabilities so the LLM, UI, and
docs can query it.  Each capability describes what it does, what it accepts
as input, what it returns, and what phrases or conditions trigger it.

Usage::

    from rex.capabilities.registry import CapabilityRegistry, get_capability_registry

    registry = get_capability_registry()
    print(registry.list())
    results = registry.search("weather")
"""

from __future__ import annotations

import builtins
import logging
import threading
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Literal

logger = logging.getLogger(__name__)


class CapabilityConflictError(ValueError):
    """Raised when duplicate capability metadata would silently diverge."""


class SecurityClassificationError(CapabilityConflictError):
    """Raised when remote metadata attempts to weaken local security metadata."""


_ALLOWED_SOURCES = {"local", "openclaw", "integration", "system"}
_ALLOWED_HEALTH = {"unknown", "healthy", "degraded", "unhealthy", "unavailable"}
_ALLOWED_OPERATIONS = {"read", "mutation"}
_ALLOWED_RISKS = {"safe", "sensitive", "prohibited"}


@dataclass
class Capability:
    """Canonical Capability / Tool Card metadata.

    ``name`` remains the compatibility spelling for the canonical ``id``. Static
    metadata is sealed after construction so runtime state updates cannot silently
    rewrite source, schemas, permissions, operation, risk, or verification policy.
    """

    name: str
    description: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    enabled: bool = True
    category: str = "General"
    integration_state: str | None = None
    read_capable: bool | None = None
    write_capable: bool | None = None
    source: str = "local"
    input_schema: dict[str, str] = field(default_factory=dict)
    output_schema: dict[str, str] = field(default_factory=dict)
    required_permissions: tuple[str, ...] = ()
    health: str = "unknown"
    operation: Literal["read", "mutation"] = "read"
    risk: Literal["safe", "sensitive", "prohibited"] = "safe"
    verification_supported: bool = False
    examples: tuple[str, ...] = ()
    requires_identity: bool = False
    required_args: tuple[str, ...] = ()
    requires_config: tuple[str, ...] = ()
    _sealed: bool = field(default=False, init=False, repr=False, compare=False)

    _STATIC_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "name",
            "description",
            "inputs",
            "outputs",
            "triggers",
            "category",
            "source",
            "input_schema",
            "output_schema",
            "required_permissions",
            "operation",
            "risk",
            "verification_supported",
            "examples",
            "requires_identity",
            "required_args",
            "requires_config",
        }
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False) and name in self._STATIC_FIELDS:
            raise AttributeError(f"Capability static metadata is sealed: {name}")
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        name = self.name.strip()
        description = self.description.strip()
        if not name:
            raise ValueError("Capability name cannot be empty")
        if not description:
            raise ValueError("Capability description cannot be empty")
        source = self.source.strip().lower()
        if source not in _ALLOWED_SOURCES:
            raise ValueError(f"Unsupported capability source: {self.source!r}")
        if self.health not in _ALLOWED_HEALTH:
            raise ValueError(f"Unsupported capability health: {self.health!r}")
        if self.operation not in _ALLOWED_OPERATIONS:
            raise ValueError(f"Unsupported capability operation: {self.operation!r}")
        if self.risk not in _ALLOWED_RISKS:
            raise ValueError(f"Unsupported capability risk: {self.risk!r}")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "source", source)
        input_schema = dict(self.input_schema) or dict.fromkeys(self.inputs, "any")
        output_schema = dict(self.output_schema) or dict.fromkeys(self.outputs, "any")
        object.__setattr__(self, "input_schema", dict(sorted(input_schema.items())))
        object.__setattr__(self, "output_schema", dict(sorted(output_schema.items())))
        object.__setattr__(
            self, "inputs", list(input_schema) if input_schema else list(self.inputs)
        )
        object.__setattr__(
            self, "outputs", list(output_schema) if output_schema else list(self.outputs)
        )
        object.__setattr__(self, "triggers", sorted(dict.fromkeys(self.triggers)))
        object.__setattr__(
            self, "required_permissions", tuple(sorted(set(self.required_permissions)))
        )
        object.__setattr__(self, "examples", tuple(dict.fromkeys(self.examples)))
        object.__setattr__(self, "required_args", tuple(dict.fromkeys(self.required_args)))
        object.__setattr__(self, "requires_config", tuple(dict.fromkeys(self.requires_config)))
        object.__setattr__(self, "_sealed", True)

    @property
    def id(self) -> str:
        """Canonical stable identifier (legacy name alias)."""
        return self.name

    def security_signature(self) -> tuple[object, ...]:
        return (
            self.operation,
            self.risk,
            self.required_permissions,
            self.requires_identity,
            self.verification_supported,
        )

    def static_signature(self) -> tuple[object, ...]:
        return (
            self.name,
            self.description,
            tuple(self.inputs),
            tuple(self.outputs),
            tuple(self.triggers),
            self.category,
            self.source,
            tuple(self.input_schema.items()),
            tuple(self.output_schema.items()),
            self.required_permissions,
            self.operation,
            self.risk,
            self.verification_supported,
            self.examples,
            self.requires_identity,
            self.required_args,
            self.requires_config,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return deterministic, serializable metadata without user-specific authority."""
        return {
            "id": self.id,
            "source": self.source,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "enabled": self.enabled,
            "required_permissions": list(self.required_permissions),
            "health": self.health,
            "operation": self.operation,
            "risk": self.risk,
            "verification_supported": self.verification_supported,
            "examples": list(self.examples),
            "category": self.category,
            "triggers": list(self.triggers),
        }


class CapabilityRegistry:
    """Central registry of all Rex capabilities.

    Capabilities are stored by name and can be listed or searched by keyword.
    The registry is typically populated at startup via
    :func:`populate_from_config`.

    Example::

        registry = CapabilityRegistry()
        registry.register(Capability(
            name="web_search",
            description="Search the web for information",
            inputs=["query"],
            outputs=["results"],
            triggers=["search", "look up", "find"],
            category="Search",
        ))
        caps = registry.list()
        matches = registry.search("search")
    """

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def register(self, capability: Capability, *, replace: bool = False) -> Capability:
        """Register canonical metadata without silent duplicate drift."""
        with self._lock:
            existing = self._capabilities.get(capability.id)
            if existing is not None:
                if existing.static_signature() == capability.static_signature():
                    return existing
                if not replace:
                    raise CapabilityConflictError(
                        f"Capability {capability.id!r} is already registered with different metadata"
                    )
            self._capabilities[capability.id] = capability
            logger.debug("Registered capability: %s", capability.id)
            return capability

    def register_remote(self, capability: Capability) -> Capability:
        """Register remote metadata without weakening an existing local card."""
        if capability.source != "openclaw":
            raise ValueError("register_remote() requires source='openclaw'")
        with self._lock:
            existing = self._capabilities.get(capability.id)
            if existing is None:
                return self.register(capability)
            if existing.security_signature() != capability.security_signature():
                raise SecurityClassificationError(
                    f"Remote metadata for {capability.id!r} conflicts with local security classification"
                )
            if existing.source != "openclaw":
                return existing
            return self.register(capability)

    def apply_openclaw_snapshot(
        self, capabilities: list[Capability] | tuple[Capability, ...]
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Atomically apply one validated OpenClaw capability snapshot.

        Remote-owned descriptive/schema metadata may refresh, but an existing
        Rex security classification is preserved. Local capabilities are never
        replaced by a remote card with the same ID. Removed OpenClaw cards stay
        visible as unavailable so stale inventory never remains executable.
        """
        desired: dict[str, Capability] = {}
        for capability in capabilities:
            if capability.source != "openclaw":
                raise ValueError("OpenClaw snapshot entries require source='openclaw'")
            if capability.id in desired:
                raise CapabilityConflictError(
                    f"Duplicate OpenClaw capability in snapshot: {capability.id!r}"
                )
            desired[capability.id] = capability

        with self._lock:
            current = self._capabilities
            staged = dict(current)
            added: list[str] = []
            updated: list[str] = []
            removed: list[str] = []

            for capability_id, incoming in desired.items():
                existing = current.get(capability_id)
                if existing is not None and existing.source != "openclaw":
                    # The local card remains the canonical authority.
                    continue
                candidate = incoming
                if existing is not None:
                    candidate = replace(
                        incoming,
                        operation=existing.operation,
                        risk=existing.risk,
                        required_permissions=existing.required_permissions,
                        requires_identity=existing.requires_identity,
                        verification_supported=existing.verification_supported,
                    )
                if existing is None:
                    added.append(capability_id)
                elif (
                    existing.static_signature() != candidate.static_signature()
                    or existing.enabled != candidate.enabled
                    or existing.health != candidate.health
                    or existing.integration_state != candidate.integration_state
                    or existing.read_capable != candidate.read_capable
                    or existing.write_capable != candidate.write_capable
                ):
                    updated.append(capability_id)
                staged[capability_id] = candidate

            desired_ids = set(desired)
            for capability_id, existing in current.items():
                if existing.source != "openclaw" or capability_id in desired_ids:
                    continue
                unavailable = replace(
                    existing,
                    enabled=False,
                    health="unavailable",
                    integration_state="unavailable",
                    read_capable=False,
                    write_capable=False,
                )
                staged[capability_id] = unavailable
                if (
                    existing.enabled
                    or existing.health != "unavailable"
                    or existing.integration_state != "unavailable"
                ):
                    removed.append(capability_id)

            self._capabilities = staged
            return tuple(sorted(added)), tuple(sorted(updated)), tuple(sorted(removed))

    def mark_openclaw_unavailable(self) -> tuple[str, ...]:
        """Atomically mark the last known OpenClaw snapshot stale/unavailable."""
        with self._lock:
            staged = dict(self._capabilities)
            changed: list[str] = []
            for capability_id, existing in self._capabilities.items():
                if existing.source != "openclaw":
                    continue
                staged[capability_id] = replace(
                    existing,
                    enabled=False,
                    health="unhealthy",
                    integration_state="unavailable",
                    read_capable=False,
                    write_capable=False,
                )
                changed.append(capability_id)
            self._capabilities = staged
            return tuple(sorted(changed))

    def update_runtime_state(self, name: str, **updates: object) -> Capability:
        """Update only mutable operational evidence for a registered card."""
        with self._lock:
            capability = self._capabilities[name]
            allowed = {"enabled", "health", "integration_state", "read_capable", "write_capable"}
            invalid = set(updates) - allowed
            if invalid:
                raise ValueError(f"Static capability metadata cannot be updated: {sorted(invalid)}")
            for field_name, value in updates.items():
                setattr(capability, field_name, value)
            if capability.health not in _ALLOWED_HEALTH:
                raise ValueError(f"Unsupported capability health: {capability.health!r}")
            return capability

    def unregister(self, name: str) -> bool:
        """Remove a capability by name.

        Args:
            name: Capability name to remove.

        Returns:
            ``True`` if the capability was found and removed, ``False`` otherwise.
        """
        with self._lock:
            if name in self._capabilities:
                del self._capabilities[name]
                logger.debug("Unregistered capability: %s", name)
                return True
            return False

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def list(self, *, include_disabled: bool = False) -> builtins.list[Capability]:
        """Return all registered capabilities, optionally including disabled ones.

        Args:
            include_disabled: When ``True``, disabled capabilities are included.

        Returns:
            List of :class:`Capability` instances sorted by name.
        """
        with self._lock:
            caps = list(self._capabilities.values())
            if not include_disabled:
                caps = [c for c in caps if c.enabled]
            return sorted(caps, key=lambda c: c.name)

    def search(self, query: str) -> builtins.list[Capability]:
        """Filter capabilities whose metadata contains *query* as a substring.

        The search is **case-insensitive** and checks the capability's
        ``name``, ``description``, ``category``, and ``triggers``.

        Args:
            query: Keyword or phrase to search for.

        Returns:
            List of matching :class:`Capability` instances (enabled only),
            sorted by name.
        """
        q = query.lower()
        with self._lock:
            results: builtins.list[Capability] = []
            for cap in self._capabilities.values():
                if not cap.enabled:
                    continue
                haystack = " ".join(
                    [cap.name, cap.description, cap.category] + cap.triggers
                ).lower()
                if q in haystack:
                    results.append(cap)
            return sorted(results, key=lambda c: c.name)

    def metadata_snapshot(self) -> builtins.list[dict[str, object]]:
        """Return a deterministic metadata snapshot sorted by stable ID."""
        return [cap.to_metadata() for cap in self.list(include_disabled=True)]

    def is_authorized(self, name: str, granted_permissions: set[str] | frozenset[str]) -> bool:
        """Evaluate authority from the caller's current permission snapshot."""
        capability = self.get(name)
        if capability is None:
            return False
        granted = set(granted_permissions)
        if "admin" in granted:
            return True
        return set(capability.required_permissions).issubset(granted)

    def get(self, name: str) -> Capability | None:
        """Look up a capability by exact name.

        Args:
            name: Capability name.

        Returns:
            The :class:`Capability` instance or ``None`` if not found.
        """
        with self._lock:
            return self._capabilities.get(name)

    def __len__(self) -> int:  # pragma: no cover
        return len(self._capabilities)


# ---------------------------------------------------------------------------
# Built-in capability definitions
# ---------------------------------------------------------------------------

_BUILTIN_CAPABILITIES: list[Capability] = [
    Capability(
        name="chat",
        description="Converse with the AI assistant about any topic",
        inputs=["user_message"],
        outputs=["assistant_reply"],
        triggers=["chat", "talk", "ask", "tell me", "what is", "explain"],
        enabled=True,
        category="General",
    ),
    Capability(
        name="time_now",
        description="Get the current date and time",
        inputs=["location"],
        outputs=["datetime_string"],
        triggers=["time", "what time", "current time", "date", "today"],
        enabled=True,
        category="General",
    ),
    Capability(
        name="weather_now",
        description="Get current weather conditions for a location",
        inputs=["location"],
        outputs=["weather_summary", "temperature", "conditions"],
        triggers=["weather", "temperature", "forecast", "how hot", "how cold", "raining"],
        enabled=False,  # enabled at startup if configured
        category="General",
    ),
    Capability(
        name="web_search",
        description="Search the web for information",
        inputs=["query"],
        outputs=["search_results"],
        triggers=["search", "look up", "google", "find online", "search for"],
        enabled=False,  # enabled at startup if configured
        category="Search",
    ),
    Capability(
        name="home_assistant",
        description="Control Home Assistant devices, scenes, and automations",
        inputs=["command", "entity_id"],
        outputs=["action_result"],
        triggers=[
            "turn on",
            "turn off",
            "lights",
            "thermostat",
            "lock",
            "unlock",
            "scene",
            "home assistant",
        ],
        enabled=False,  # enabled at startup if configured
        category="Home",
    ),
    Capability(
        name="send_email",
        description="Compose and send email messages",
        inputs=["to", "subject", "body"],
        outputs=["send_status"],
        triggers=["send email", "email", "write email", "compose email"],
        enabled=False,  # enabled at startup if configured
        category="Communication",
    ),
    Capability(
        name="music_assistant",
        description="Play music via Music Assistant",
        inputs=["query", "action"],
        outputs=["playback_status"],
        triggers=["play music", "play", "pause", "stop music", "next song", "music"],
        enabled=False,  # enabled at startup if configured
        category="Entertainment",
    ),
]


def _build_default_registry() -> CapabilityRegistry:
    """Create a registry pre-loaded with built-in capabilities (all disabled by default)."""
    registry = CapabilityRegistry()
    executable_tool_ids = {"time_now", "weather_now", "web_search", "send_email"}
    for cap in _BUILTIN_CAPABILITIES:
        if cap.name not in executable_tool_ids:
            registry.register(cap)
    return registry


def populate_from_config(registry: CapabilityRegistry, config: object) -> None:
    """Enable capabilities in *registry* based on the provided *config*.

    Inspects ``AppConfig`` attributes (or any object with the same attribute
    names) and enables the corresponding capability when the relevant
    integration appears to be configured.

    Args:
        registry: :class:`CapabilityRegistry` to update in-place.
        config: An ``AppConfig`` instance (or any compatible object).
    """

    def _has(attr: str) -> bool:
        val = getattr(config, attr, None)
        return bool(val)

    # weather
    cap = registry.get("weather_now")
    if cap is not None:
        cap.enabled = _has("openweathermap_api_key")

    from rex.integration_state import build_integration_inventory

    evidence = {item.key: item for item in build_integration_inventory(config)}

    def apply_integration_state(capability_name: str, integration_key: str) -> None:
        capability = registry.get(capability_name)
        item = evidence.get(integration_key)
        if capability is None or item is None:
            return
        capability.enabled = item.available and item.configured
        capability.integration_state = item.state.value
        capability.read_capable = item.read_capable
        capability.write_capable = item.write_capable

    # web search
    apply_integration_state("web_search", "search")

    # Home Assistant
    apply_integration_state("home_assistant", "home_assistant")

    # email
    apply_integration_state("send_email", "email")

    # music assistant
    cap = registry.get("music_assistant")
    if cap is not None:
        cap.enabled = _has("music_assistant_url")

    logger.debug(
        "Capability registry populated: %d enabled",
        sum(1 for c in registry.list(include_disabled=False)),
    )


# ---------------------------------------------------------------------------
# Global registry singleton
# ---------------------------------------------------------------------------

_registry: CapabilityRegistry | None = None


def get_capability_registry(config: object | None = None) -> CapabilityRegistry:
    """Return the global :class:`CapabilityRegistry`, creating it if needed.

    On first call the registry is built from built-in capabilities.  If
    *config* is provided it is used to enable integrations via
    :func:`populate_from_config`.

    Args:
        config: Optional ``AppConfig``-compatible object.  When given the
            registry is (re-)populated from the config values.

    Returns:
        The global :class:`CapabilityRegistry` singleton.
    """
    global _registry
    if _registry is None:
        _registry = _build_default_registry()
        from rex.tools.registry import ensure_default_registry  # noqa: PLC0415

        ensure_default_registry(_registry)
        logger.debug("CapabilityRegistry created with %d canonical capabilities", len(_registry))
    if config is not None:
        populate_from_config(_registry, config)
    return _registry


def reset_capability_registry() -> None:
    """Reset the global registry (useful in tests)."""
    global _registry
    _registry = None
    try:
        from rex.tools.registry import reset_default_registry  # noqa: PLC0415

        reset_default_registry()
    except ImportError:
        pass


__all__ = [
    "Capability",
    "CapabilityConflictError",
    "CapabilityRegistry",
    "SecurityClassificationError",
    "get_capability_registry",
    "populate_from_config",
    "reset_capability_registry",
]
