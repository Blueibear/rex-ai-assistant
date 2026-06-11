"""Messaging (msg, notify) commands for the Rex CLI.

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


def cmd_msg(args: argparse.Namespace) -> int:
    """Manage messaging."""
    # OPENCLAW-REPLACE: cmd_msg stub — SMS backend is being retired.
    # Programmatic SMS access is available via rex.openclaw.tools.sms_tool.send_sms.
    subcommand = args.msg_command

    if subcommand == "send":
        channel = getattr(args, "channel", "").lower()
        if channel == "sms":
            print("SMS messaging not available (migrating to OpenClaw messaging backend)")
            return 1
        print(f"Error: Unsupported channel '{channel}'. Currently only 'sms' is supported.")
        return 1

    if subcommand == "receive":
        channel = getattr(args, "channel", "").lower()
        if channel == "sms":
            print("SMS messaging not available (migrating to OpenClaw messaging backend)")
            return 1
        print(f"Error: Unsupported channel '{channel}'. Currently only 'sms' is supported.")
        return 1

    print("Unknown messaging subcommand. Use 'rex msg --help'")
    return 1


def cmd_notify(args: argparse.Namespace) -> int:
    """Manage notifications."""
    from rex.notification import NotificationRequest, get_escalation_manager, get_notifier

    user_id = _cli()._resolve_cli_user(args)
    notifier = get_notifier()
    escalation_manager = get_escalation_manager()
    subcommand = args.notify_command

    if subcommand == "send":
        if args.channels:
            channel_list = [ch.strip() for ch in args.channels.split(",")]
        else:
            channel_list = ["dashboard"]

        metadata: dict = {}
        if user_id:
            metadata["user_id"] = user_id

        notification = NotificationRequest(
            priority=args.priority,
            title=args.title,
            body=args.body,
            channel_preferences=channel_list,
            metadata=metadata,
        )

        notifier.send(notification)

        if notification.priority == "urgent":
            next_channel = channel_list[1] if len(channel_list) > 1 else "email"
            escalation_manager.track_notification(notification, next_channel)

        print("Notification sent successfully")
        print(f"  ID: {notification.id}")
        print(f"  Priority: {notification.priority}")
        print(f"  Channels: {', '.join(channel_list)}")
        print(f"  Title: {notification.title}")
        return 0

    if subcommand == "list-digests":
        digests = notifier.list_digests()

        if not digests or all(len(q) == 0 for q in digests.values()):
            print("No queued digest notifications.")
            return 0

        print("Queued Digest Notifications")
        print("=" * 80)
        print()

        for channel, notifications in digests.items():
            if not notifications:
                continue

            print(f"Channel: {channel}")
            print(f"  Count: {len(notifications)}")
            print()

            for notif in notifications:
                print(f"  - {notif['id']}: {notif['title']}")
                print(f"    Created: {notif['timestamp']}")
                body_preview = (
                    notif["body"][:60] + "..." if len(notif["body"]) > 60 else notif["body"]
                )
                print(f"    Body: {body_preview}")
                print()

        total_count = sum(len(q) for q in digests.values())
        print(f"Total: {total_count} queued notifications across {len(digests)} channels")
        return 0

    if subcommand == "flush-digests":
        channel = args.channel
        count = notifier.flush_digests(channel=channel)

        if count == 0:
            print("No digest notifications to flush.")
        else:
            if channel:
                print(f"Flushed digest queue for channel: {channel}")
            else:
                print(f"Flushed {count} digest queue(s)")
        return 0

    if subcommand == "ack":
        notification_id = args.notification_id
        if escalation_manager.acknowledge(notification_id):
            print(f"Acknowledged notification: {notification_id}")
            return 0
        print(f"Notification not found or already acknowledged: {notification_id}")
        return 1

    print("Unknown notification subcommand. Use 'rex notify --help'")
    return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register this domain's subcommands on the top-level subparsers."""
    # msg (messaging)
    msg_parser = subparsers.add_parser(
        "msg",
        help="Send and receive messages (SMS via Twilio or stub)",
        description="Manage messaging through various channels (SMS, Telegram, etc.). Real SMS delivery requires Twilio credentials configured via CredentialManager; defaults to stub/mock mode for offline development.",
    )
    msg_parser.add_argument(
        "--user",
        type=str,
        help="Override the active user identity",
    )
    msg_parser.add_argument(
        "--account-id",
        type=str,
        help="Messaging account ID to use (overrides default)",
        dest="account_id",
    )
    msg_subparsers = msg_parser.add_subparsers(
        title="messaging commands",
        dest="msg_command",
        metavar="COMMAND",
    )

    msg_send = msg_subparsers.add_parser(
        "send",
        help="Send a message",
        description="Send a message via a specific channel.",
    )
    msg_send.add_argument(
        "--channel", type=str, default="sms", help="Channel to send via (default: sms)"
    )
    msg_send.add_argument(
        "--to", type=str, required=True, help="Recipient (phone number, user ID, etc.)"
    )
    msg_send.add_argument("--body", type=str, required=True, help="Message body text")
    msg_send.set_defaults(func=_cli().cmd_msg, msg_command="send")

    msg_receive = msg_subparsers.add_parser(
        "receive",
        help="Receive recent messages",
        description="List recent inbound messages from a channel.",
    )
    msg_receive.add_argument(
        "--channel", type=str, default="sms", help="Channel to receive from (default: sms)"
    )
    msg_receive.add_argument(
        "--limit", type=int, default=10, help="Maximum number of messages (default: 10)"
    )
    msg_receive.set_defaults(func=_cli().cmd_msg, msg_command="receive")

    msg_parser.set_defaults(func=_cli().cmd_msg, msg_command="receive")

    # notify (notifications)
    notify_parser = subparsers.add_parser(
        "notify",
        help="Send and manage notifications",
        description="Multi-channel notification system with priority routing.",
    )
    notify_parser.add_argument(
        "--user",
        type=str,
        help="Override the active user identity",
    )
    notify_subparsers = notify_parser.add_subparsers(
        title="notification commands",
        dest="notify_command",
        metavar="COMMAND",
    )

    notify_send = notify_subparsers.add_parser(
        "send",
        help="Send a notification",
        description="Send a notification with priority and channel preferences.",
    )
    notify_send.add_argument(
        "--priority",
        type=str,
        default="normal",
        choices=["urgent", "normal", "digest"],
        help="Priority level (default: normal)",
    )
    notify_send.add_argument("--title", type=str, required=True, help="Notification title")
    notify_send.add_argument("--body", type=str, required=True, help="Notification body")
    notify_send.add_argument(
        "--channels", type=str, help="Comma-separated list of channels (e.g., sms,email,dashboard)"
    )
    notify_send.set_defaults(func=_cli().cmd_notify, notify_command="send")

    notify_list_digests = notify_subparsers.add_parser(
        "list-digests",
        help="List queued digest notifications",
        description="Show all notifications queued for digest delivery.",
    )
    notify_list_digests.set_defaults(func=_cli().cmd_notify, notify_command="list-digests")

    notify_flush = notify_subparsers.add_parser(
        "flush-digests",
        help="Flush digest queues",
        description="Send all queued digest notifications immediately.",
    )
    notify_flush.add_argument(
        "--channel", type=str, help="Specific channel to flush (default: all)"
    )
    notify_flush.set_defaults(func=_cli().cmd_notify, notify_command="flush-digests")

    notify_ack = notify_subparsers.add_parser(
        "ack",
        help="Acknowledge a notification",
        description="Mark a notification as acknowledged (prevents escalation).",
    )
    notify_ack.add_argument("notification_id", type=str, help="Notification ID to acknowledge")
    notify_ack.set_defaults(func=_cli().cmd_notify, notify_command="ack")

    notify_parser.set_defaults(func=_cli().cmd_notify, notify_command="list-digests")
