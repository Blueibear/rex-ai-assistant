"""Reminders and follow-up cues commands for the Rex CLI.

Extracted verbatim from ``rex/cli.py`` (US-REM-027). Handler behavior,
argument definitions, help text, defaults, and exit codes are unchanged.
"""

from __future__ import annotations

import argparse

from rex.commands._helpers import _parse_datetime_strict


def _cli():
    """Return the ``rex.cli`` module at call time.

    ``rex.cli`` is the single patch point for service getters and command
    handlers (tests monkeypatch ``rex.cli.<name>``). Resolving through the
    module at call time preserves that behavior without creating an import
    cycle at module load time.
    """
    from rex import cli

    return cli


def cmd_reminders(args: argparse.Namespace) -> int:
    """Manage reminders."""
    service = _cli().get_reminder_service()
    subcommand = args.reminders_command

    if subcommand == "add":
        title = args.title
        at_str = args.at

        try:
            remind_at = _parse_datetime_strict(at_str)
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1

        followup = bool(getattr(args, "followup", False))

        try:
            from rex.config_manager import load_config

            config = load_config()
            user_id = config.get("runtime", {}).get("user_id", "default")
        except Exception:
            user_id = "default"

        # Compatibility: service might implement create_reminder(...) or add_reminder(...)
        if hasattr(service, "create_reminder"):
            reminder = service.create_reminder(
                user_id=user_id,
                title=title,
                remind_at=remind_at,
                followup_enabled=followup,
            )
        else:
            reminder = service.add_reminder(title, remind_at, follow_up=followup)

        reminder_id = getattr(reminder, "reminder_id", getattr(reminder, "id", "unknown"))
        remind_at_val = getattr(reminder, "remind_at", getattr(reminder, "remind_at", remind_at))
        if hasattr(remind_at_val, "isoformat"):
            remind_at_str = remind_at_val.isoformat()
        else:
            remind_at_str = str(remind_at_val)

        print(f"Created reminder: {reminder_id}")
        print(f"  Title: {getattr(reminder, 'title', title)}")
        print(f"  Remind at: {remind_at_str}")
        print(f"  Follow-up enabled: {getattr(reminder, 'followup_enabled', followup)}")
        return 0

    if subcommand == "list":
        status_filter = getattr(args, "status", None)
        reminders = service.list_reminders(status=status_filter)

        if not reminders:
            print("No reminders found.")
            return 0

        print("Reminders")
        print("=" * 60)
        print()

        for rem in reminders:
            status_indicator = {
                "pending": "[PENDING]",
                "fired": "[FIRED]",
                "done": "[DONE]",
                "canceled": "[CANCELED]",
            }.get(getattr(rem, "status", getattr(rem, "status", "unknown")), "[?]")

            rid = getattr(rem, "reminder_id", getattr(rem, "id", "unknown"))
            title = getattr(rem, "title", "")
            remind_at_dt = getattr(rem, "remind_at", getattr(rem, "remind_at", None))
            remind_at_str = remind_at_dt.strftime("%Y-%m-%d %H:%M") if remind_at_dt else "n/a"

            print(f"{rid}: {title} {status_indicator}")
            print(f"  Remind at: {remind_at_str}")

            if getattr(rem, "followup_enabled", False):
                print("  Follow-up: enabled")

            fired_at = getattr(rem, "fired_at", None)
            done_at = getattr(rem, "done_at", None)
            if fired_at:
                print(f"  Fired at: {fired_at.strftime('%Y-%m-%d %H:%M')}")
            if done_at:
                print(f"  Done at: {done_at.strftime('%Y-%m-%d %H:%M')}")
            print()

        print(f"Total: {len(reminders)} reminders")
        return 0

    if subcommand == "done":
        reminder_id = getattr(args, "reminder_id", None) or getattr(args, "id", None)
        if hasattr(service, "mark_done"):
            ok = service.mark_done(reminder_id)
        else:
            ok = service.complete_reminder(reminder_id)
        if ok:
            print(f"Marked reminder {reminder_id} as done.")
            return 0
        print(f"Error: Reminder not found: {reminder_id}")
        return 1

    if subcommand == "cancel":
        reminder_id = getattr(args, "reminder_id", None) or getattr(args, "id", None)
        if hasattr(service, "cancel_reminder"):
            ok = service.cancel_reminder(reminder_id)
        else:
            ok = service.cancel(reminder_id)
        if ok:
            print(f"Canceled reminder {reminder_id}.")
            return 0
        print(f"Error: Reminder not found: {reminder_id}")
        return 1

    print("Unknown reminders subcommand. Use 'rex reminders --help'")
    return 1


def cmd_cues(args: argparse.Namespace) -> int:
    """Manage follow-up cues."""
    store = _cli().get_cue_store()
    subcommand = args.cues_command

    if subcommand == "list":
        status_filter = getattr(args, "status", None)

        if hasattr(store, "list_all_cues"):
            cues = store.list_all_cues(status=status_filter)
        else:
            cues = store.list_cues(status=status_filter)

        if not cues:
            print("No cues found.")
            return 0

        print("Follow-up Cues")
        print("=" * 60)
        print()

        for cue in cues:
            status_indicator = {
                "pending": "[PENDING]",
                "asked": "[ASKED]",
                "dismissed": "[DISMISSED]",
            }.get(getattr(cue, "status", "unknown"), "[?]")

            source_type = getattr(cue, "source_type", None)
            source_label = f"[{source_type}]" if source_type else ""
            cue_id = getattr(cue, "cue_id", getattr(cue, "id", "unknown"))
            title = getattr(cue, "title", "(no title)")
            prompt = getattr(cue, "prompt", "")

            print(f"{cue_id}: {title} {source_label} {status_indicator}".strip())
            print(f"  Prompt: {prompt}")

            created_at = getattr(cue, "created_at", None)
            expires_at = getattr(cue, "expires_at", None)
            asked_at = getattr(cue, "asked_at", None)
            dismissed_at = getattr(cue, "dismissed_at", None)

            if created_at:
                print(f"  Created: {created_at.strftime('%Y-%m-%d %H:%M')}")
            if expires_at:
                print(f"  Expires: {expires_at.strftime('%Y-%m-%d %H:%M')}")
            if asked_at:
                print(f"  Asked at: {asked_at.strftime('%Y-%m-%d %H:%M')}")
            if dismissed_at:
                print(f"  Dismissed at: {dismissed_at.strftime('%Y-%m-%d %H:%M')}")
            print()

        print(f"Total: {len(cues)} cues")

        if hasattr(store, "stats"):
            stats = store.stats()
            by_status = stats.get("by_status", {}) if isinstance(stats, dict) else {}
            print(f"  Pending: {by_status.get('pending', 0)}")
            print(f"  Asked: {by_status.get('asked', 0)}")
            print(f"  Dismissed: {by_status.get('dismissed', 0)}")

        return 0

    if subcommand == "dismiss":
        cue_id = args.cue_id
        ok = store.dismiss(cue_id) if hasattr(store, "dismiss") else False
        if ok:
            print(f"Dismissed cue {cue_id}.")
            return 0
        print(f"Error: Cue not found: {cue_id}")
        return 1

    if subcommand == "prune":
        # Compatibility: store may implement prune_expired() or prune_expired(expire_hours=...)
        if hasattr(store, "prune_expired"):
            try:
                count = store.prune_expired()
            except TypeError:
                from rex import settings

                expire_hours = int(getattr(settings, "followups_expire_hours", 168))
                count = store.prune_expired(expire_hours=expire_hours)
        else:
            count = 0
        print(f"Pruned {count} expired cue(s).")
        return 0

    print("Unknown cues subcommand. Use 'rex cues --help'")
    return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register this domain's subcommands on the top-level subparsers."""
    # reminders
    reminders_parser = subparsers.add_parser(
        "reminders",
        help="Manage reminders",
        description="Create and manage one-off reminders with optional follow-up cues.",
    )
    reminders_subparsers = reminders_parser.add_subparsers(
        title="reminders commands",
        dest="reminders_command",
        metavar="COMMAND",
    )

    reminders_add = reminders_subparsers.add_parser(
        "add",
        help="Add a new reminder",
        description="Create a new one-off reminder at a specific date/time.",
    )
    reminders_add.add_argument("title", type=str, help="Reminder title/description")
    reminders_add.add_argument(
        "--at", type=str, required=True, help="When to remind (YYYY-MM-DD HH:MM)"
    )
    reminders_add.add_argument(
        "--follow-up",
        dest="followup",
        action="store_true",
        help="Create a follow-up cue after reminder fires",
    )
    reminders_add.set_defaults(func=_cli().cmd_reminders, reminders_command="add")

    reminders_list = reminders_subparsers.add_parser(
        "list",
        help="List reminders",
        description="List all reminders with their status.",
    )
    reminders_list.add_argument(
        "--status",
        type=str,
        choices=["pending", "fired", "done", "canceled"],
        help="Filter by status",
    )
    reminders_list.set_defaults(func=_cli().cmd_reminders, reminders_command="list")

    reminders_done = reminders_subparsers.add_parser(
        "done",
        help="Mark a reminder as done",
        description="Mark a reminder as completed.",
    )
    reminders_done.add_argument("id", type=str, help="Reminder ID to mark as done")
    reminders_done.set_defaults(func=_cli().cmd_reminders, reminders_command="done")

    reminders_cancel = reminders_subparsers.add_parser(
        "cancel",
        help="Cancel a reminder",
        description="Cancel a pending reminder.",
    )
    reminders_cancel.add_argument("id", type=str, help="Reminder ID to cancel")
    reminders_cancel.set_defaults(func=_cli().cmd_reminders, reminders_command="cancel")

    reminders_parser.set_defaults(func=_cli().cmd_reminders, reminders_command="list")

    # cues
    cues_parser = subparsers.add_parser(
        "cues",
        help="Manage follow-up cues",
        description="View and manage follow-up cues for conversations.",
    )
    cues_subparsers = cues_parser.add_subparsers(
        title="cues commands",
        dest="cues_command",
        metavar="COMMAND",
    )

    cues_list = cues_subparsers.add_parser(
        "list",
        help="List cues",
        description="List all follow-up cues with their status.",
    )
    cues_list.add_argument(
        "--status", type=str, choices=["pending", "asked", "dismissed"], help="Filter by status"
    )
    cues_list.set_defaults(func=_cli().cmd_cues, cues_command="list")

    cues_dismiss = cues_subparsers.add_parser(
        "dismiss",
        help="Dismiss a cue",
        description="Dismiss a follow-up cue so it won't be asked.",
    )
    cues_dismiss.add_argument("cue_id", type=str, help="Cue ID to dismiss")
    cues_dismiss.set_defaults(func=_cli().cmd_cues, cues_command="dismiss")

    cues_prune = cues_subparsers.add_parser(
        "prune",
        help="Prune expired cues",
        description="Remove all expired cues from the store.",
    )
    cues_prune.set_defaults(func=_cli().cmd_cues, cues_command="prune")

    cues_parser.set_defaults(func=_cli().cmd_cues, cues_command="list")
