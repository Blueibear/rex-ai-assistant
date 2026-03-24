"""OpenClaw config bridge — maps Rex settings to OpenClaw agent configuration.

Translates ``rex.config.AppConfig`` fields into the configuration dict that
OpenClaw's agent registration API expects.  Because OpenClaw's exact config
schema is an open dependency (see PRD Section 8.3), the output is a plain
``dict`` that will be refined once the API surface is confirmed.

Typical usage::

    from rex.openclaw.config import build_agent_config

    agent_config = build_agent_config()          # uses global Rex config
    # or:
    from rex.config import load_config
    agent_config = build_agent_config(load_config())
"""

from __future__ import annotations

import dataclasses
from typing import Any

from rex.config import AppConfig, load_config
from rex.profile_manager import DEFAULT_PROFILES_DIR, load_profile


def build_system_prompt(config: AppConfig | None = None) -> str:
    """Build a Rex persona system prompt from ``AppConfig`` fields.

    The prompt encodes Rex's identity, active profile, location context, and
    capability set so that any LLM invoked through the OpenClaw bridge has a
    consistent persona regardless of how it was initialised.

    Args:
        config: Rex ``AppConfig`` instance.  When ``None`` the global config
            is loaded via ``rex.config.load_config()``.

    Returns:
        A system-prompt string suitable for the ``system`` role message.
    """
    if config is None:
        config = load_config()

    name = config.wakeword.capitalize()
    parts: list[str] = [f"You are {name}, a helpful and friendly AI assistant."]

    profile = config.active_profile
    if profile and profile != "default":
        parts.append(f"Your active profile is: {profile}.")

    if config.default_location:
        parts.append(f"The user's location is {config.default_location}.")

    if config.default_timezone:
        parts.append(f"The local timezone is {config.default_timezone}.")

    if config.capabilities:
        caps = ", ".join(config.capabilities)
        parts.append(f"Your capabilities include: {caps}.")

    return " ".join(parts)


def apply_profile_to_config(
    config: AppConfig,
    profile_name: str,
    profiles_dir: str = DEFAULT_PROFILES_DIR,
) -> AppConfig:
    """Apply a named Rex profile to an ``AppConfig`` and return a new instance.

    Loads *profile_name* from *profiles_dir*, replaces the ``capabilities``
    list wholesale (as :func:`rex.profile_manager.apply_profile` specifies),
    and sets ``active_profile`` to *profile_name*.  All other ``AppConfig``
    fields are preserved unchanged.

    Args:
        config: Base ``AppConfig`` instance to apply the profile to.
        profile_name: Name of the profile (without ``.json`` extension) to
            load from *profiles_dir*.
        profiles_dir: Directory that contains ``<name>.json`` profile files.
            Defaults to ``"profiles"``.

    Returns:
        A new ``AppConfig`` with ``capabilities`` and ``active_profile``
        updated from the loaded profile.

    Raises:
        FileNotFoundError: If the profile file does not exist.
        ValueError: If the profile fails schema validation.
    """
    profile = load_profile(profile_name, profiles_dir)
    capabilities: list[str] = [c for c in profile.get("capabilities", []) if c]
    return dataclasses.replace(
        config,
        capabilities=capabilities,
        active_profile=profile_name,
    )


def build_agent_config(config: AppConfig | None = None) -> dict[str, Any]:
    """Map Rex ``AppConfig`` fields to an OpenClaw agent configuration dict.

    The returned dict captures the subset of Rex settings that are relevant
    to an OpenClaw agent:

    - **identity** — agent name, user ID, active profile
    - **llm** — provider, model, generation parameters
    - **context** — location, timezone
    - **memory** — max conversation turns, memory byte budget
    - **persona** — wake word (used as the agent's canonical name)

    .. note::
        The exact keys expected by OpenClaw are not yet confirmed (PRD §8.3).
        Keys prefixed with ``rex_`` are Rex-specific and will be remapped once
        the OpenClaw config schema is known.  Do not rely on specific key names
        until that open dependency is resolved.

    Args:
        config: Rex ``AppConfig`` instance.  When ``None`` the global config
            is loaded via ``rex.config.load_config()``.

    Returns:
        A plain ``dict`` suitable for passing to OpenClaw's agent registration
        or configuration API.
    """
    if config is None:
        config = load_config()

    return {
        # Agent identity
        "agent_name": config.wakeword.capitalize(),  # "Rex"
        "user_id": config.user_id,
        "active_profile": config.active_profile,
        # LLM / generation settings
        "llm_provider": config.llm_provider,
        "llm_model": config.llm_model,
        "llm_temperature": config.llm_temperature,
        "llm_top_p": config.llm_top_p,
        "llm_top_k": config.llm_top_k,
        "llm_max_tokens": config.llm_max_tokens,
        # Context
        "default_location": config.default_location,
        "default_timezone": config.default_timezone,
        # Memory / conversation limits
        "memory_max_turns": config.memory_max_turns,
        "memory_max_bytes": config.memory_max_bytes,
        # Rex-specific extras (remapped once OpenClaw API is confirmed)
        "rex_capabilities": list(config.capabilities),
        "rex_tts_provider": config.tts_provider,
        "rex_speak_language": config.speak_language,
    }
