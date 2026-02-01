"""Async CLI loop driving the full Rex voice experience.

This preserves the semantics of the original ``rex_loop.py`` entry point while
bridging to the refactored voice loop package introduced during previous
iterations.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import warnings
from collections.abc import Iterable

# Suppress torio FFmpeg extension warnings (non-critical audio codec features)
warnings.filterwarnings("ignore", message=".*FFmpeg extension.*")
warnings.filterwarnings("ignore", message=".*libtorio.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="torio")

logger = logging.getLogger(__name__)


def _select_plugins(enabled: Iterable[str] | None):
    from rex.plugins import PluginSpec, load_plugins

    specs = load_plugins()
    if not enabled:
        return specs
    enabled_set = {name.strip() for name in enabled if name}
    return [spec for spec in specs if spec.name in enabled_set]


async def _run(args) -> None:
    import rex
    from rex.assistant import Assistant
    from rex.assistant_errors import AssistantError, ConfigurationError, WakeWordError
    from rex.config import load_config as load_runtime_config
    from rex.logging_utils import configure_logging
    from rex.plugins import shutdown_plugins
    from rex.voice_loop import build_voice_loop

    # MQTT is optional
    try:
        from rex.mqtt_audio_router import MqttAudioRouter

        MQTT_AVAILABLE = True
    except ImportError as exc:
        MqttAudioRouter = None  # type: ignore
        MQTT_AVAILABLE = False
        _mqtt_import_error = str(exc)
    configure_logging()

    # Run migration from legacy .env to rex_config.json if needed
    from rex.config_manager import migrate_legacy_env_to_config, get_legacy_env_warnings
    migration_notes = migrate_legacy_env_to_config()
    if migration_notes and len(migration_notes) > 1:
        logger.info("Configuration migration completed")

    # Warn about legacy environment variables
    legacy_warnings = get_legacy_env_warnings()
    if legacy_warnings:
        logger.warning("Legacy environment variables detected. These are now ignored. Use config/rex_config.json instead.")

    try:
        runtime_config = load_runtime_config(reload=True)
        rex.settings = runtime_config
    except ConfigurationError as exc:
        logger.error("Profile configuration error: %s", exc)
        return

    plugin_specs = _select_plugins(args.enable_plugin)

    if args.user:
        os.environ["REX_ACTIVE_USER"] = args.user
        rex.reload_settings()

    assistant = Assistant(history_limit=rex.settings.max_memory_items, plugins=plugin_specs)

    try:
        voice_loop = build_voice_loop(assistant)
    except (AssistantError, WakeWordError) as exc:
        logger.error("Unable to initialise voice loop: %s", exc)
        return

    mqtt_router = None
    if MQTT_AVAILABLE and MqttAudioRouter is not None:
        try:
            mqtt_router = MqttAudioRouter(assistant=assistant)
            await mqtt_router.start()
            logger.info("MQTT audio router started.")
        except Exception as exc:  # pragma: no cover - defensive startup log
            logger.error("Unable to start MQTT audio router: %s", exc)
            mqtt_router = None
    elif not MQTT_AVAILABLE:
        logger.info("MQTT audio router disabled (dependency not available)")

    logger.info("🎙️ Voice loop started. Press Ctrl+C to exit.")
    try:
        await voice_loop.run()
    finally:
        if mqtt_router is not None:
            await mqtt_router.stop()
        shutdown_plugins(plugin_specs)


def main(argv: list[str] | None = None) -> int:
    from utils.env_loader import load as _load_env

    _load_env()

    parser = argparse.ArgumentParser(description="Run the Rex voice assistant loop.")
    parser.add_argument("--user", help="Override the active user profile")
    parser.add_argument(
        "--enable-plugin",
        action="append",
        metavar="NAME",
        help="Explicitly enable a plugin by name (omit to load all)",
    )

    args = parser.parse_args(argv)

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
