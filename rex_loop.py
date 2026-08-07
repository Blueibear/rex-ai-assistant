"""Async CLI loop driving the full Rex voice experience.

This preserves the semantics of the original ``rex_loop.py`` entry point while
bridging to the refactored voice loop package introduced during previous
iterations.
"""

# ruff: noqa: E402

from __future__ import annotations

# Load .env before accessing any environment variables
from utils.env_loader import load as _load_env

_load_env()

import argparse
import asyncio
import logging
import os
import warnings
from collections.abc import Iterable
from pathlib import Path

# Suppress torio FFmpeg extension warnings (non-critical audio codec features)
warnings.filterwarnings("ignore", message=".*FFmpeg extension.*")
warnings.filterwarnings("ignore", message=".*libtorio.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="torio")

import rex
from rex.assistant import Assistant
from rex.assistant_errors import AssistantError, ConfigurationError, WakeWordError
from rex.config import load_config as load_runtime_config
from rex.logging_utils import configure_logging
from rex.plugins import PluginSpec, load_plugins, shutdown_plugins
from rex.voice_loop import build_voice_loop

# MQTT is optional
try:
    from rex.mqtt_audio_router import MqttAudioRouter

    MQTT_AVAILABLE = True
except ImportError as exc:
    MqttAudioRouter = None  # type: ignore
    MQTT_AVAILABLE = False
    _mqtt_import_error = str(exc)

logger = logging.getLogger(__name__)

# Backward compatibility: re-export AsyncRexAssistant for code that imports it from here
# The canonical location is voice_loop.py at the repo root
try:
    from voice_loop import AsyncRexAssistant
    from voice_loop import build_voice_loop as _build_voice_loop_v1
except ImportError:
    AsyncRexAssistant = None  # type: ignore
    _build_voice_loop_v1 = None  # type: ignore


def _select_plugins(enabled: Iterable[str] | None) -> list[PluginSpec]:
    specs = load_plugins()
    if not enabled:
        return specs
    enabled_set = {name.strip() for name in enabled if name}
    return [spec for spec in specs if spec.name in enabled_set]


async def _run(args) -> None:
    configure_logging()

    # Run migration from legacy .env to rex_config.json if needed
    from rex.config_manager import get_legacy_env_warnings, migrate_legacy_env_to_config

    migration_notes = migrate_legacy_env_to_config()
    if migration_notes and len(migration_notes) > 1:
        logger.info("Configuration migration completed")

    # Warn about legacy environment variables
    legacy_warnings = get_legacy_env_warnings()
    if legacy_warnings:
        logger.warning(
            "Legacy environment variables detected. These are now ignored. Use config/rex_config.json instead."
        )

    try:
        runtime_config = load_runtime_config(reload=True)
        rex.settings = runtime_config
        configure_logging()
    except ConfigurationError as exc:
        logger.error("Profile configuration error: %s", exc)
        return

    plugin_specs = _select_plugins(args.enable_plugin)

    if args.user:
        os.environ["REX_ACTIVE_USER"] = args.user
        rex.reload_settings()

    # Deliberate single-user profile selection (issue #303): --user wins,
    # then the identify/session chain, then the configured profile, then the
    # explicit "default" profile.  Assistant no longer invents an identity.
    from rex.identity import resolve_entrypoint_user_id

    user_id = resolve_entrypoint_user_id(rex.settings, explicit_user=args.user)
    assistant = Assistant(
        history_limit=rex.settings.max_memory_items, plugins=plugin_specs, user_id=user_id
    )

    try:
        wake_sound_path = getattr(rex.settings, "wake_sound_path", None)
        voice_loop = build_voice_loop(
            assistant,
            activation_mode=args.mode,
            sample_rate=rex.settings.sample_rate,
            detection_seconds=rex.settings.detection_frame_seconds,
            capture_seconds=rex.settings.capture_seconds,
            whisper_model=rex.settings.whisper_model,
            device=rex.settings.whisper_device,
            language=rex.settings.whisper_language or "en",
            wake_sound_path=Path(wake_sound_path) if wake_sound_path else None,
        )
    except (AssistantError, WakeWordError) as exc:
        logger.error("Unable to initialise voice loop: %s", exc)
        return

    mqtt_router = None
    mqtt_broker = getattr(rex.settings, "mqtt_broker", None)
    if MQTT_AVAILABLE and MqttAudioRouter is not None and mqtt_broker:
        try:
            mqtt_router = MqttAudioRouter(assistant=assistant)
            await mqtt_router.start()
            logger.info("MQTT audio router started.")
        except Exception as exc:  # pragma: no cover - defensive startup log
            logger.error("Unable to start MQTT audio router: %s", exc)
            mqtt_router = None
    elif not MQTT_AVAILABLE:
        logger.info("MQTT audio router disabled (dependency not available)")
    else:
        logger.info("MQTT audio router disabled (mqtt_broker not configured)")

    logger.info("🎙️ Voice loop started. Press Ctrl+C to exit.")
    try:
        await voice_loop.run()
    finally:
        if mqtt_router is not None:
            await mqtt_router.stop()
        shutdown_plugins(plugin_specs)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AskRex voice assistant loop.")
    parser.add_argument(
        "--mode",
        choices=("hold-to-talk", "wake-word"),
        default="hold-to-talk",
        help=(
            "Voice activation mode (default: hold-to-talk). "
            "Use wake-word to opt into the beta wake detector."
        ),
    )
    parser.add_argument("--user", help="Override the active user profile")
    parser.add_argument(
        "--enable-plugin",
        action="append",
        metavar="NAME",
        help="Explicitly enable a plugin by name (omit to load all)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _create_parser().parse_args(argv)

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
