"""Email and calendar commands for the Rex CLI.

Extracted verbatim from ``rex/cli.py`` (US-REM-027). Handler behavior,
argument definitions, help text, defaults, and exit codes are unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _cli():
    """Return the ``rex.cli`` module at call time.

    ``rex.cli`` is the single patch point for service getters and command
    handlers (tests monkeypatch ``rex.cli.<name>``). Resolving through the
    module at call time preserves that behavior without creating an import
    cycle at module load time.
    """
    from rex import cli

    return cli


_NO_USER_MSG = (
    "Error: No active user for email.\n"
    "Set one with: rex identify --user <id>\n"
    "Or pass one explicitly: rex email ... --user <id>"
)


def _resolve_email_user(args: argparse.Namespace) -> str | None:
    """Resolve the requesting user for email commands, failing closed.

    Uses ``--user`` when given, otherwise the standard identity chain
    (``rex identify`` session state, then ``runtime.active_user`` /
    ``runtime.user_id`` in config). Never silently falls back to the
    ``default`` profile — single-user setups select it explicitly with
    ``--user default`` or ``rex identify --user default``.
    """
    try:
        user = _cli()._resolve_cli_user(args)
        return str(user) if user else None
    except ValueError as exc:
        print(f"Error: {exc}")
        return None


def cmd_email(args: argparse.Namespace) -> int:
    """Manage email.  Every subcommand runs as one validated user."""
    from rex.assistant_errors import IntegrationNotConfiguredError

    user = _resolve_email_user(args)
    if not user:
        print(_NO_USER_MSG)
        return 1

    subcommand = args.email_command

    if subcommand == "accounts":
        return _cmd_email_accounts(args, user)

    if subcommand == "test-connection":
        return _cmd_email_test_connection(args, user)

    try:
        email_service = _cli().get_email_service()
    except IntegrationNotConfiguredError:
        print("Email integration not configured. Set IMAP/SMTP credentials in config.")
        return 1

    if subcommand == "unread":
        if not email_service.connected:
            if not email_service.connect(user):
                print("Error: Failed to connect to email service")
                return 1

        limit = getattr(args, "limit", None) or 10
        try:
            unread = email_service.fetch_unread(limit=limit, user_id=user)
        except PermissionError as exc:
            print(f"Error: {exc}")
            return 1
        print("Unread Email Summary")
        print("=" * 80)
        print()

        if not unread:
            print("No unread emails.")
            return 0

        for email in unread:
            category = email_service.categorize(email)
            importance = "!! " if getattr(email, "importance_score", 0.0) >= 0.8 else ""
            print(f"{importance}{email.id}: {email.subject}")
            print(f"  From: {email.from_addr}")
            print(f"  Received: {email.received_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Category: {category}")
            if getattr(args, "verbose", False):
                score = getattr(email, "importance_score", None)
                if score is not None:
                    print(f"  Importance: {score:.2f}")
                print(f"  Snippet: {email.snippet}")
            print()

        print(f"Total: {len(unread)} unread emails")
        return 0

    if subcommand == "send":
        return _cmd_email_send(args, user)

    print("Unknown email subcommand. Use 'rex email --help'")
    return 1


def _cmd_email_accounts(args: argparse.Namespace, user: str) -> int:
    """Handle 'rex email accounts' subcommands for one validated user."""
    accounts_cmd = getattr(args, "accounts_command", "list")
    resolver = _cli()._load_email_resolver_safe()

    if accounts_cmd == "list":
        accounts = [] if resolver is None else resolver.accounts_for_user(user)
        if resolver is None or not accounts:
            print(f"No email accounts configured for user '{user}'.")
            print()
            print("To configure accounts, add an 'email' section to config/rex_config.json")
            print(f"and assign them to this user under users.{user}.email_accounts.")
            print("See docs/email.md for details.")
            return 0

        print(f"Email Accounts for user '{user}'")
        print("=" * 60)
        print()

        default_id = resolver.default_account_id_for_user(user)
        for acct in accounts:
            is_default = " (default)" if acct.id == default_id else ""
            print(f"  {acct.id}{is_default}")
            if acct.label:
                print(f"    Label:   {acct.label}")
            print(f"    Address: {acct.address}")
            print(f"    IMAP:    {acct.imap.host}:{acct.imap.port} (SSL={acct.imap.ssl})")
            print(f"    SMTP:    {acct.smtp.host}:{acct.smtp.port} (STARTTLS={acct.smtp.starttls})")
            print(f"    Cred:    {acct.credential_ref}")
            print()

        print(f"Total: {len(accounts)} account(s)")
        return 0

    if accounts_cmd == "set-active":
        account_id = getattr(args, "account_id", None)
        if not account_id:
            print("Error: --account-id is required")
            return 1

        if resolver is None:
            print("Error: No email configuration found")
            return 1

        owned = resolver.account_ids_for_user(user)
        if account_id not in owned:
            # Foreign and nonexistent accounts are indistinguishable; only
            # the requesting user's own accounts are ever revealed.
            print(f"Error: Account '{account_id}' is not available for user '{user}'.")
            if owned:
                print(f"Available: {', '.join(owned)}")
            return 1

        # Update only this user's default; never another user's routing and
        # never the legacy global default.
        try:
            import json as _json

            config_path = Path("config/rex_config.json")
            if config_path.exists():
                config_data = _json.loads(config_path.read_text(encoding="utf-8"))
            else:
                config_data = {}

            users_block = config_data.setdefault("users", {})
            user_entry = users_block.setdefault(user, {})
            if not isinstance(user_entry, dict):
                print("Error: Invalid users section in config")
                return 1
            user_entry["default_email_account_id"] = account_id
            config_path.write_text(_json.dumps(config_data, indent=2) + "\n", encoding="utf-8")
            print(f"Default email account for user '{user}' set to '{account_id}'")
            return 0
        except Exception as exc:
            print(f"Error updating config: {exc}")
            return 1

    print("Unknown accounts subcommand. Use 'rex email accounts --help'")
    return 1


def _cmd_email_send(args: argparse.Namespace, user: str) -> int:
    """Handle 'rex email send' for one validated user."""
    to = getattr(args, "to", None)
    subject = getattr(args, "subject", None)
    body = getattr(args, "body", None)
    account_id = getattr(args, "account_id", None)

    if not to or not subject or not body:
        print("Error: --to, --subject, and --body are required")
        return 1

    from rex.assistant_errors import IntegrationNotConfiguredError

    try:
        email_service = _cli().get_email_service()
    except IntegrationNotConfiguredError:
        print("Email integration not configured. Set IMAP/SMTP credentials in config.")
        return 1
    if not email_service.connected:
        if not email_service.connect(user):
            print("Error: Failed to connect to email service")
            return 1

    try:
        result = email_service.send(
            to=to,
            subject=subject,
            body=body,
            account_id=account_id,
            user_id=user,
        )
    except PermissionError as exc:
        print(f"Error: {exc}")
        return 1

    if result.get("ok"):
        msg_id = result.get("message_id") or "(stub)"
        print(f"Email sent successfully (message_id: {msg_id})")
        return 0

    print(f"Error: {result.get('error', 'Unknown error')}")
    return 1


def _cmd_email_test_connection(args: argparse.Namespace, user: str) -> int:
    """Handle 'rex email test-connection' for one validated user."""
    account_id = getattr(args, "account_id", None)

    resolver = _cli()._load_email_resolver_safe()
    if resolver is None or not resolver.has_configured_accounts():
        print("No email accounts configured. Using stub backend.")
        print("Connection test: OK (stub mode)")
        return 0

    owned = resolver.account_ids_for_user(user)
    try:
        resolved_id = resolver.resolve_account_id(user, account_id)
    except PermissionError:
        resolved_id = None
    acct = resolver.get_account_definition(resolved_id) if resolved_id else None
    if acct is None:
        # Only the requesting user's own accounts are ever revealed.
        if owned:
            print(f"Error: Account not available. Available for user '{user}': {', '.join(owned)}")
        else:
            print(f"Error: No email accounts configured for user '{user}'.")
        return 1

    print(f"Testing connection for account '{acct.id}' ({acct.address})...")
    print(f"  IMAP: {acct.imap.host}:{acct.imap.port}")
    print(f"  SMTP: {acct.smtp.host}:{acct.smtp.port}")

    # Check credentials (only this account's own credential_ref is consulted)
    try:
        from rex.credentials import get_credential_manager

        cm = get_credential_manager()
        token = cm.get_token(acct.credential_ref)
        if not token:
            print(f"  Credential ({acct.credential_ref}): NOT FOUND")
            print()
            print("Error: No credentials available for this account.")
            print(f"Set the environment variable for '{acct.credential_ref}' or")
            print("add it to config/credentials.json.")
            return 1
        print(f"  Credential ({acct.credential_ref}): available")
    except Exception as exc:
        print(f"  Credential check error: {exc}")
        return 1

    # Try IMAP connect
    try:
        from rex.email_backends.account_router import build_backend_for_account

        backend = build_backend_for_account(acct, credential_getter=cm.get_token)
        if backend is not None and backend.connect():
            print("  IMAP connection: OK")
            backend.disconnect()
        else:
            print("  IMAP connection: FAILED")
            return 1
    except Exception as exc:
        print(f"  IMAP connection error: {exc}")
        return 1

    print()
    print("Connection test passed.")
    return 0


def cmd_calendar(args: argparse.Namespace) -> int:
    """Manage calendar."""
    _ = _cli()._resolve_cli_user(args)
    calendar_service = _cli().get_calendar_service()
    subcommand = args.calendar_command

    if subcommand == "test-connection":
        return _cmd_calendar_test_connection()

    if subcommand == "upcoming":
        if not calendar_service.connected:
            if not calendar_service.connect():
                print("Error: Failed to connect to calendar service")
                return 1

        days = getattr(args, "days", None) or 7
        events = calendar_service.get_upcoming_events(days=days)

        print(f"Upcoming Events (next {days} days)")
        print("=" * 80)
        print()

        if not events:
            print(f"No upcoming events in the next {days} days.")
            return 0

        for event in events:
            if getattr(event, "all_day", False):
                time_str = event.start_time.strftime("%Y-%m-%d") + " (All day)"
            else:
                time_str = (
                    event.start_time.strftime("%Y-%m-%d %H:%M")
                    + " - "
                    + event.end_time.strftime("%H:%M")
                )

            print(f"{event.id}: {event.title}")
            print(f"  When: {time_str}")
            if event.location:
                print(f"  Location: {event.location}")
            if getattr(event, "attendees", None):
                if event.attendees:
                    print(f"  Attendees: {', '.join(event.attendees)}")
            if getattr(args, "verbose", False) and getattr(event, "description", None):
                print(f"  Description: {event.description}")
            print()

        print(f"Total: {len(events)} events")

        if getattr(args, "conflicts", False):
            conflicts = calendar_service.find_conflicts(events)
            if conflicts:
                print()
                print("Conflicts Detected:")
                print("-" * 80)
                for event1, event2 in conflicts:
                    print(f"!! '{event1.title}' overlaps with '{event2.title}'")

        return 0

    print("Unknown calendar subcommand. Use 'rex calendar --help'")
    return 1


def _cmd_calendar_test_connection() -> int:
    """Verify calendar backend configuration."""
    from rex.calendar_backends.factory import create_calendar_backend

    backend = create_calendar_backend()
    ok, message = backend.test_connection()
    name = backend.backend_name

    if ok:
        print(f"Calendar backend '{name}': OK")
        if message:
            print(f"  {message}")
        return 0
    else:
        print(f"Calendar backend '{name}': FAILED")
        if message:
            print(f"  {message}")
        return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register this domain's subcommands on the top-level subparsers."""
    # email
    email_parser = subparsers.add_parser(
        "email",
        help="Manage email (supports real IMAP/SMTP and stub mode)",
        description="Read, send, and triage emails. Supports real IMAP/SMTP backends when configured, or stub/mock data for offline development.",
    )
    email_parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="User context for this command (overrides session/config active user)",
    )
    email_subparsers = email_parser.add_subparsers(
        title="email commands",
        dest="email_command",
        metavar="COMMAND",
    )

    email_unread = email_subparsers.add_parser(
        "unread",
        help="Fetch unread emails",
        description="Display unread emails with categorization.",
    )
    email_unread.add_argument(
        "--limit", type=int, default=10, help="Maximum number of emails to fetch (default: 10)"
    )
    email_unread.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed email information"
    )
    email_unread.add_argument(
        "--user", type=str, default=None, help="User context for this command"
    )
    email_unread.set_defaults(func=_cli().cmd_email, email_command="unread")

    # email accounts
    email_accounts = email_subparsers.add_parser(
        "accounts",
        help="Manage email accounts",
        description="List and manage configured email accounts.",
    )
    email_accounts_sub = email_accounts.add_subparsers(
        title="accounts commands",
        dest="accounts_command",
        metavar="COMMAND",
    )

    email_accounts_list = email_accounts_sub.add_parser(
        "list",
        help="List the selected user's email accounts",
    )
    email_accounts_list.add_argument(
        "--user", type=str, default=None, help="User context for this command"
    )
    email_accounts_list.set_defaults(
        func=cmd_email, email_command="accounts", accounts_command="list"
    )

    email_accounts_set_active = email_accounts_sub.add_parser(
        "set-active",
        help="Set the selected user's default email account",
    )
    email_accounts_set_active.add_argument(
        "--account-id",
        type=str,
        required=True,
        help="Account ID to set as this user's default",
    )
    email_accounts_set_active.add_argument(
        "--user", type=str, default=None, help="User context for this command"
    )
    email_accounts_set_active.set_defaults(
        func=cmd_email, email_command="accounts", accounts_command="set-active"
    )

    email_accounts.set_defaults(
        func=_cli().cmd_email, email_command="accounts", accounts_command="list"
    )

    # email send
    email_send = email_subparsers.add_parser(
        "send",
        help="Send an email",
        description="Send an email via the configured backend.",
    )
    email_send.add_argument("--to", type=str, required=True, help="Recipient email address")
    email_send.add_argument("--subject", type=str, required=True, help="Email subject")
    email_send.add_argument("--body", type=str, required=True, help="Email body text")
    email_send.add_argument("--account-id", type=str, default=None, help="Account ID to send from")
    email_send.add_argument("--user", type=str, default=None, help="User context for this command")
    email_send.set_defaults(func=_cli().cmd_email, email_command="send")

    # email test-connection
    email_test_conn = email_subparsers.add_parser(
        "test-connection",
        help="Test email account connection",
        description="Verify IMAP/SMTP connectivity for a configured account.",
    )
    email_test_conn.add_argument("--account-id", type=str, default=None, help="Account ID to test")
    email_test_conn.add_argument(
        "--user", type=str, default=None, help="User context for this command"
    )
    email_test_conn.set_defaults(func=_cli().cmd_email, email_command="test-connection")

    email_parser.set_defaults(func=_cli().cmd_email, email_command="unread")

    # calendar
    calendar_parser = subparsers.add_parser(
        "calendar",
        help="Manage calendar (ICS read-only backend available + stub fallback)",
        description="View and manage calendar events. Supports ICS read-only backend or stub/mock data.",
    )
    calendar_parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="User context for this command (overrides session/config active user)",
    )
    calendar_subparsers = calendar_parser.add_subparsers(
        title="calendar commands",
        dest="calendar_command",
        metavar="COMMAND",
    )

    calendar_upcoming = calendar_subparsers.add_parser(
        "upcoming",
        help="Show upcoming events",
        description="Display upcoming calendar events.",
    )
    calendar_upcoming.add_argument(
        "--days", type=int, default=7, help="Number of days to look ahead (default: 7)"
    )
    calendar_upcoming.add_argument(
        "--conflicts", action="store_true", help="Check for scheduling conflicts"
    )
    calendar_upcoming.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed event information"
    )
    calendar_upcoming.set_defaults(func=_cli().cmd_calendar, calendar_command="upcoming")

    calendar_test_conn = calendar_subparsers.add_parser(
        "test-connection",
        help="Verify calendar backend configuration",
        description="Check that the configured calendar backend can connect and parse events.",
    )
    calendar_test_conn.set_defaults(func=_cli().cmd_calendar, calendar_command="test-connection")

    calendar_parser.set_defaults(func=_cli().cmd_calendar, calendar_command="upcoming")
