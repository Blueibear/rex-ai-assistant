"""Home Assistant commands for the Rex CLI.

Extracted verbatim from ``rex/cli.py`` (US-REM-027). Handler behavior,
argument definitions, help text, defaults, and exit codes are unchanged.
"""

from __future__ import annotations

import argparse


def _cli():
    """Return the ``rex.cli`` module at call time.

    ``rex.cli`` is the single patch point for service getters and command
    handlers (tests monkeypatch ``rex.cli.<name>``). Resolving through the
    module at call time preserves that behavior without creating an import
    cycle at module load time.
    """
    from rex import cli

    return cli


def cmd_ha(args: argparse.Namespace) -> int:
    """Home Assistant integration commands."""
    ha_command = getattr(args, "ha_command", None)
    if ha_command == "tts":
        return _cmd_ha_tts(args)
    if ha_command == "approve":
        return _cmd_ha_approve(args)
    print("Unknown ha subcommand. Use 'rex ha --help'")
    return 1


def _cmd_ha_approve(args: argparse.Namespace) -> int:
    """List discovered HA devices and interactively approve or ignore them."""
    from rex.config import load_config
    from rex.ha.device_aliases import AliasResolver
    from rex.ha.discovery import (
        approve_device,
        discover_devices,
        ignore_device,
        load_ignored_devices,
    )

    cfg = load_config()
    base_url: str = getattr(cfg, "ha_base_url", "") or ""
    token: str = getattr(cfg, "ha_token", "") or ""

    if not base_url or not token:
        print("Home Assistant is not configured (ha_base_url / HA_TOKEN not set).")
        print(
            "Set these values in config/rex_config.json and your .env before running 'rex ha approve'."
        )
        return 1

    print("Discovering devices from Home Assistant…")
    devices = discover_devices(base_url=base_url, token=token)
    if not devices:
        print("No devices found.")
        return 0

    # Load already-known entity IDs from aliases file and ignore file
    aliases_path = getattr(args, "aliases_path", None)
    ignore_path = getattr(args, "ignore_path", None)

    resolver = AliasResolver(aliases_path)
    approved_entity_ids: set[str] = set(resolver._aliases.values())  # noqa: SLF001
    ignored_entity_ids: set[str] = set(load_ignored_devices(ignore_path))

    pending = [
        d
        for d in devices
        if d["entity_id"] not in approved_entity_ids and d["entity_id"] not in ignored_entity_ids
    ]

    if not pending:
        print(f"All {len(devices)} discovered device(s) are already approved or ignored.")
        return 0

    print(f"\nFound {len(pending)} pending device(s):\n")
    for i, dev in enumerate(pending, 1):
        print(f"  [{i:3d}] {dev['entity_id']}  ({dev['friendly_name']}, state={dev['state']})")

    print(
        "\nFor each device enter:\n"
        "  <alias>   — approve with this friendly name\n"
        "  i         — ignore (hide from Rex)\n"
        "  s / Enter — skip for now\n"
    )

    approved_count = 0
    ignored_count = 0
    for dev in pending:
        entity_id = dev["entity_id"]
        friendly = dev["friendly_name"]
        try:
            response = input(f"  {entity_id} ({friendly}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            break

        if response.lower() == "i":
            ignore_device(entity_id, ignore_path)
            print(f"    → ignored {entity_id}")
            ignored_count += 1
        elif response and response.lower() not in ("s", "skip"):
            approve_device(entity_id, response, aliases_path)
            print(f"    → approved as '{response}'")
            approved_count += 1
        else:
            print("    → skipped")

    print(f"\nDone: {approved_count} approved, {ignored_count} ignored.")
    return 0


def _cmd_ha_tts(args: argparse.Namespace) -> int:
    """Handle 'rex ha tts <subcommand>'."""
    tts_command = getattr(args, "ha_tts_command", None)
    if tts_command == "test":
        return _cmd_ha_tts_test(args)
    print("Unknown ha tts subcommand. Use 'rex ha tts --help'")
    return 1


def _cmd_ha_tts_test(args: argparse.Namespace) -> int:
    """Send a test TTS announcement via the Home Assistant channel."""
    from rex.ha_tts.config import load_ha_tts_config

    cfg = load_ha_tts_config()

    if not cfg.enabled:
        print("HA TTS channel: disabled")
        print("  Set notifications.ha_tts.enabled=true in config/rex_config.json to enable.")
        return 0

    if not cfg.base_url or not cfg.token_ref:
        print("HA TTS channel: configured but incomplete")
        if not cfg.base_url:
            print("  Missing: notifications.ha_tts.base_url")
        if not cfg.token_ref:
            print("  Missing: notifications.ha_tts.token_ref")
        return 1

    entity_id = getattr(args, "entity_id", None) or cfg.default_entity_id
    if not entity_id:
        print("Error: no entity_id specified and no default_entity_id configured.")
        print(
            "  Pass --entity-id <entity> or set notifications.ha_tts.default_entity_id in config."
        )
        return 1

    message = args.message

    print("HA TTS channel: sending test announcement")
    print(f"  base_url : {cfg.base_url}")
    print(f"  entity_id: {entity_id}")
    print(f"  message  : {message!r}")

    from rex.ha_tts.client import build_ha_tts_client

    client = build_ha_tts_client()
    if client is None:
        print("Error: could not build HA TTS client (check logs for details).")
        return 1

    result = client.speak(message, entity_id=entity_id)
    if result.ok:
        print("OK: announcement sent successfully.")
        return 0
    print(f"Error: {result.error}")
    return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register this domain's subcommands on the top-level subparsers."""
    # ha (Home Assistant integration commands)
    ha_parser = subparsers.add_parser(
        "ha",
        help="Home Assistant integration commands",
        description=(
            "Commands for interacting with the Home Assistant integration.\n\n"
            "The HA TTS channel must be configured in config/rex_config.json\n"
            "under notifications.ha_tts before use."
        ),
    )
    ha_subparsers = ha_parser.add_subparsers(
        title="ha commands",
        dest="ha_command",
        metavar="COMMAND",
    )

    # ha tts
    ha_tts_parser = ha_subparsers.add_parser(
        "tts",
        help="Home Assistant TTS commands",
        description="Commands for the Home Assistant TTS notification channel.",
    )
    ha_tts_subparsers = ha_tts_parser.add_subparsers(
        title="ha tts commands",
        dest="ha_tts_command",
        metavar="COMMAND",
    )

    # ha tts test
    ha_tts_test = ha_tts_subparsers.add_parser(
        "test",
        help="Send a test TTS announcement via Home Assistant",
        description=(
            "Send a test TTS announcement to verify the Home Assistant TTS\n"
            "channel is configured correctly.  Requires notifications.ha_tts\n"
            "to be enabled in config/rex_config.json."
        ),
    )
    ha_tts_test.add_argument(
        "--message",
        type=str,
        default="Rex Home Assistant TTS test announcement.",
        help="Text to announce (default: test message)",
    )
    ha_tts_test.add_argument(
        "--entity-id",
        dest="entity_id",
        type=str,
        default=None,
        help=(
            "Target media player entity ID " "(overrides notifications.ha_tts.default_entity_id)"
        ),
    )
    ha_tts_test.set_defaults(func=_cli().cmd_ha, ha_command="tts", ha_tts_command="test")

    ha_tts_parser.set_defaults(func=_cli().cmd_ha, ha_command="tts", ha_tts_command="test")

    # ha approve
    ha_approve_parser = ha_subparsers.add_parser(
        "approve",
        help="List discovered HA devices and approve or ignore them",
        description=(
            "Scan Home Assistant for devices, then interactively approve each one\n"
            "with a friendly alias or mark it as ignored.  Approved devices are\n"
            "written to config/device_aliases.json; ignored devices to\n"
            "config/device_ignore.json."
        ),
    )
    ha_approve_parser.add_argument(
        "--aliases-path",
        dest="aliases_path",
        type=str,
        default=None,
        help="Override path to device_aliases.json (default: config/device_aliases.json)",
    )
    ha_approve_parser.add_argument(
        "--ignore-path",
        dest="ignore_path",
        type=str,
        default=None,
        help="Override path to device_ignore.json (default: config/device_ignore.json)",
    )
    ha_approve_parser.set_defaults(func=_cli().cmd_ha, ha_command="approve")

    ha_parser.set_defaults(func=_cli().cmd_ha, ha_command="tts")
