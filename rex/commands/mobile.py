"""Mobile API gateway commands for the Rex CLI (issue #323).

Commands:

- ``rex mobile-api [--host HOST] [--port PORT]`` — run the authenticated
  mobile API development server.
- ``rex mobile-user create --username USERNAME`` — safely create a mobile
  user reusing the canonical user store, profile, and permission bootstrap.

Heavy imports (Flask, the mobile package) happen inside the handlers so
``--help`` and module import stay lightweight.
"""

from __future__ import annotations

import argparse
import ipaddress

_LOOPBACK_HOSTS = {"localhost"}


def _host_is_loopback(host: str) -> bool:
    """Return True when *host* only accepts local connections."""
    if host in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def cmd_mobile_api(args: argparse.Namespace) -> int:
    """Run the mobile API development server."""
    from rex.config import load_config
    from rex.mobile_api.app import create_mobile_app
    from rex.mobile_api.auth import MobileAuthConfigurationError

    settings = load_config()
    config = settings.mobile_api

    # Precedence: explicit CLI flags > typed mobile_api config > safe defaults
    # (the typed config already carries the safe localhost defaults).
    host = args.host if args.host is not None else config.host
    port = args.port if args.port is not None else config.port
    if not 1 <= port <= 65535:
        print(f"Error: invalid port {port}; must be between 1 and 65535.")
        return 1
    effective = config.model_copy(update={"host": host, "port": port})

    try:
        app = create_mobile_app(config=effective)
    except MobileAuthConfigurationError as exc:
        print(f"Error: {exc}")
        return 1

    print("AskRex mobile API gateway")
    print(f"  Bind:       {host}:{port}")
    print(f"  Status URL: http://{host}:{port}/mobile/status")
    print(f"  API version: {effective.api_version}")
    print(f"  TLS expected upstream: {'yes' if effective.require_tls else 'no'}")
    if not _host_is_loopback(host) and not effective.require_tls:
        print(
            "  WARNING: binding beyond loopback without TLS. This is a "
            "development-only configuration for trusted local networks; "
            "authentication and rate limiting stay enforced, but traffic "
            "is not encrypted. Do not expose this bind to the internet."
        )
    print("  Press Ctrl+C to stop.")

    app.run(host=host, port=port)
    return 0


def _create_mobile_user(username: str, password: str) -> dict:
    """Create the user, matching profile, and first-user admin bootstrap.

    Separated so tests can exercise the creation flow without prompts.
    """
    from rex.auth import create_user
    from rex.identity import create_user_profile
    from rex.mobile_api.db import default_users_db_path, migrate_users_db
    from rex.permissions import bootstrap_admin_if_first_user, get_permissions

    # Ensure the canonical schema (including mobile tables and the
    # user-active column) exists before touching the store.
    migrate_users_db(default_users_db_path())

    user = create_user(username, password)
    try:
        create_user_profile(user["id"], name=username)
    except FileExistsError:
        # A profile keyed by this fresh UUID should not exist; if it somehow
        # does, keep the existing profile rather than overwriting it.
        pass
    bootstrap_admin_if_first_user(user["id"])
    user["permissions"] = get_permissions(user["id"])
    return user


def cmd_mobile_user(args: argparse.Namespace) -> int:
    """Manage mobile users (currently: create)."""
    if args.mobile_user_command != "create":
        print("Error: unknown mobile-user command. Use: rex mobile-user create")
        return 1

    import getpass

    username = args.username.strip()
    if not username:
        print("Error: --username must not be empty.")
        return 1

    # Passwords are prompted with getpass (twice), never accepted via argv
    # and never echoed or logged.  An interrupted prompt creates nothing.
    try:
        password = getpass.getpass("Password: ")
        confirm = getpass.getpass("Confirm password: ")
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled. No user was created.")
        return 1

    if not password:
        print("Error: password must not be empty. No user was created.")
        return 1
    if password != confirm:
        print("Error: passwords do not match. No user was created.")
        return 1

    try:
        user = _create_mobile_user(username, password)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Created mobile user '{user['username']}'.")
    print(f"  User ID: {user['id']}")
    if "admin" in user.get("permissions", []):
        print("  Granted: admin (first registered user)")
    print("  Log in from the mobile app with this username and password.")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register mobile gateway subcommands on the top-level subparsers."""
    # mobile-api
    api_parser = subparsers.add_parser(
        "mobile-api",
        help="Run the authenticated mobile API gateway (development server)",
        description=(
            "Run the AskRex mobile API gateway.\n\n"
            "Defaults to 127.0.0.1:8765. Binding to 0.0.0.0 is an explicit,\n"
            "development-only choice for trusted local networks and prints a\n"
            "warning when TLS is not expected upstream.\n\n"
            "Requires REX_JWT_SECRET in .env (at least 32 characters)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    api_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Bind host (default: mobile_api.host config, then 127.0.0.1)",
    )
    api_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port (default: mobile_api.port config, then 8765)",
    )
    api_parser.set_defaults(func=cmd_mobile_api)

    # mobile-user
    user_parser = subparsers.add_parser(
        "mobile-user",
        help="Manage mobile users (create)",
        description=(
            "Manage users for the mobile API gateway.\n\n"
            "Users are stored in the canonical data/users.db with bcrypt\n"
            "password hashes. The first registered user is granted admin."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    user_subparsers = user_parser.add_subparsers(
        title="mobile-user commands",
        dest="mobile_user_command",
        metavar="COMMAND",
        required=True,
    )
    create_parser = user_subparsers.add_parser(
        "create",
        help="Create a mobile user (prompts for the password securely)",
        description=(
            "Create a mobile user. The password is prompted twice with\n"
            "getpass and is never accepted on the command line, echoed,\n"
            "or logged. A generated UUID is the canonical user ID."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    create_parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="Username for login (display only; not an authorization key)",
    )
    create_parser.set_defaults(func=cmd_mobile_user)


__all__ = [
    "cmd_mobile_api",
    "cmd_mobile_user",
    "register",
]
