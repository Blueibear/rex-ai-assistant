"""Interactive-session Voice Agent for the persistent Rex background runtime."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from rex.assistant_errors import AudioDeviceError, TextToSpeechError, WakeWordError
from rex.background.core_client import CoreAssistantProxy, CoreClient
from rex.background.paths import BackgroundPaths
from rex.background.types import ComponentHealth, HealthState
from rex.identity import resolve_active_user, validate_user_id
from rex.voice_loop import build_voice_loop


class _CoreUnavailable(RuntimeError):
    """Internal marker for unreadable or invalid local Core endpoint metadata."""


@dataclass(frozen=True, slots=True)
class VoiceAgentRuntime:
    """Constructed Voice Agent components before the audio loop is run."""

    proxy: CoreAssistantProxy
    loop: Any


def build_voice_agent(
    user_id: str,
    paths: BackgroundPaths,
    *,
    activation_mode: str = "wake-word",
    origin_device_id: str | None = None,
) -> VoiceAgentRuntime:
    """Build the canonical voice loop with Core as the Assistant implementation."""

    fallback_user_id = validate_user_id(user_id)
    if activation_mode not in {"hold-to-talk", "wake-word"}:
        raise ValueError(f"Unsupported voice activation mode: {activation_mode!r}")

    try:
        client = CoreClient.from_endpoint_file(paths.core_endpoint_file)
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError) as exc:
        raise _CoreUnavailable from exc

    proxy = CoreAssistantProxy(
        client=client,
        user_id=fallback_user_id,
        user_resolver=resolve_active_user,
        origin_device_id=origin_device_id,
    )
    loop = build_voice_loop(proxy, activation_mode=activation_mode)
    return VoiceAgentRuntime(proxy=proxy, loop=loop)


async def run_voice_agent(
    user_id: str,
    paths: BackgroundPaths,
    *,
    activation_mode: str = "wake-word",
    origin_device_id: str | None = None,
) -> ComponentHealth:
    """Run one Voice Agent lifecycle and return content-free terminal health."""

    # Validate caller configuration before mapping runtime failures so a bad
    # configured profile never masquerades as a transient Core/audio outage.
    validate_user_id(user_id)
    if activation_mode not in {"hold-to-talk", "wake-word"}:
        raise ValueError(f"Unsupported voice activation mode: {activation_mode!r}")

    try:
        runtime = build_voice_agent(
            user_id,
            paths,
            activation_mode=activation_mode,
            origin_device_id=origin_device_id,
        )
    except _CoreUnavailable:
        return _health(HealthState.DEGRADED, "core_unavailable")
    except AudioDeviceError:
        return _health(HealthState.UNAVAILABLE, "microphone_unavailable")
    except TextToSpeechError:
        return _health(HealthState.UNAVAILABLE, "speaker_unavailable")
    except WakeWordError:
        return _health(HealthState.UNAVAILABLE, "wakeword_unavailable")

    await runtime.loop.run()
    return _health(HealthState.STOPPED, None)


def _health(state: HealthState, detail_code: str | None) -> ComponentHealth:
    return ComponentHealth(
        component="voice_agent",
        state=state,
        detail_code=detail_code,
        observed_at=time.time(),
        pid=os.getpid(),
    )
