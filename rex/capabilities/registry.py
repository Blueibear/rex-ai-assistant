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
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Capability:
    """Metadata describing a single Rex capability.

    Attributes:
        name: Unique slug for the capability (e.g. ``"home_assistant"``).
        description: Human-readable description of what the capability does.
        inputs: List of input parameter names / types the capability accepts.
        outputs: List of output value names / types the capability returns.
        triggers: Words or phrases that typically invoke this capability.
        enabled: Whether the capability is currently available for use.
        category: Grouping label used by the UI (e.g. "Home", "Search").
    """

    name: str
    description: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    enabled: bool = True
    category: str = "General"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Capability name cannot be empty")
        if not self.description:
            raise ValueError("Capability description cannot be empty")


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

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def register(self, capability: Capability) -> None:
        """Add or overwrite a capability in the registry.

        Args:
            capability: :class:`Capability` instance to register.
        """
        self._capabilities[capability.name] = capability
        logger.debug("Registered capability: %s", capability.name)

    def unregister(self, name: str) -> bool:
        """Remove a capability by name.

        Args:
            name: Capability name to remove.

        Returns:
            ``True`` if the capability was found and removed, ``False`` otherwise.
        """
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
        results: builtins.list[Capability] = []
        for cap in self._capabilities.values():
            if not cap.enabled:
                continue
            haystack = " ".join([cap.name, cap.description, cap.category] + cap.triggers).lower()
            if q in haystack:
                results.append(cap)
        return sorted(results, key=lambda c: c.name)

    def get(self, name: str) -> Capability | None:
        """Look up a capability by exact name.

        Args:
            name: Capability name.

        Returns:
            The :class:`Capability` instance or ``None`` if not found.
        """
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
    for cap in _BUILTIN_CAPABILITIES:
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

    # web search
    cap = registry.get("web_search")
    if cap is not None:
        has_search = _has("brave_api_key") or bool(
            getattr(config, "search_providers", "duckduckgo")
        )
        cap.enabled = has_search

    # Home Assistant
    cap = registry.get("home_assistant")
    if cap is not None:
        cap.enabled = _has("ha_token") or _has("ha_base_url")

    # email
    cap = registry.get("send_email")
    if cap is not None:
        provider = getattr(config, "email_provider", "none")
        accounts = getattr(config, "email_accounts", [])
        cap.enabled = (provider != "none") or bool(accounts)

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
        logger.debug("CapabilityRegistry created with %d built-in capabilities", len(_registry))
    if config is not None:
        populate_from_config(_registry, config)
    return _registry


def reset_capability_registry() -> None:
    """Reset the global registry (useful in tests)."""
    global _registry
    _registry = None


__all__ = [
    "Capability",
    "CapabilityRegistry",
    "get_capability_registry",
    "populate_from_config",
    "reset_capability_registry",
]
