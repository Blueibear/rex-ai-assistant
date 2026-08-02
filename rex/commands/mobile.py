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


def cmd_mobile_api(args: argparse.Namespace) -> int:
    """Run the mobile API gateway.

    Any non-loopback bind requires usable TLS (S7): ``create_mobile_app``
    fails closed with ``MobileTlsConfigurationError`` before a socket is ever
    opened if TLS material cannot be provisioned. Loopback binds stay plain
    HTTP for local development unless ``mobile_api.require_tls`` opts in.
    """
    from rex.config import load_config
    from rex.mobile_api.app import create_mobile_app
    from rex.mobile_api.auth import MobileAuthConfigurationError
    from rex.mobile_api.tls import MobileTlsConfigurationError, host_is_loopback

    settings = load_config()
    config = settings.mobile_api

    # Precedence: explicit CLI flags > typed mobile_api config > safe defaults
    # (the typed config already carries the safe localhost defaults).
    host = args.host if args.host is not None else config.host
    port = args.port if args.port is not None else config.port
    if not 1 <= port <= 65535:
        print(f"Error: invalid port {port}; must be between 1 and 65535.")
        return 1
    effective = config.model_copy(
        update={
            "host": host,
            "port": port,
            "advertised_host": (
                getattr(args, "advertised_host", None)
                if getattr(args, "advertised_host", None) is not None
                else config.advertised_host
            ),
            "advertised_port": (
                getattr(args, "advertised_port", None)
                if getattr(args, "advertised_port", None) is not None
                else config.advertised_port
            ),
        }
    )

    try:
        app = create_mobile_app(config=effective)
    except MobileAuthConfigurationError as exc:
        print(f"Error: {exc}")
        return 1
    except MobileTlsConfigurationError as exc:
        print(f"Error: {exc}")
        if not host_is_loopback(host):
            print(
                "  Non-loopback binds require usable TLS and cannot start "
                "without it. Bind to 127.0.0.1/localhost for local "
                "development, or resolve the TLS provisioning error above "
                "(see docs/mobile/MOBILE_API_SETUP_WINDOWS.md)."
            )
        return 1

    tls_material = app.extensions.get("mobile_api_tls")
    scheme = "https" if tls_material else "http"
    print("AskRex mobile API gateway")
    print(f"  Bind:       {host}:{port}")
    services = app.extensions.get("mobile_api_services")
    binding = getattr(services, "transport_binding", None)
    status_origin = binding.server_url if binding is not None else f"{scheme}://{host}:{port}"
    print(f"  Status URL: {status_origin}/mobile/status")
    print(f"  API version: {effective.api_version}")
    if tls_material is not None:
        print("  TLS: enabled (desktop-owned self-signed certificate)")
        print(f"  Certificate fingerprint (SHA-256): {tls_material.fingerprint_sha256}")
        print(
            f"  SPKI pin (SHA-256/base64): "
            f"{getattr(tls_material, 'spki_pin_sha256_b64', 'unavailable')}"
        )
        print("  Pair mobile devices by QR before they connect.")
    else:
        print("  TLS: disabled (loopback development bind)")
    print("  Press Ctrl+C to stop.")

    run_kwargs: dict = {"host": host, "port": port}
    if tls_material is not None:
        run_kwargs["ssl_context"] = tls_material.build_ssl_context()
    app.run(**run_kwargs)
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
            "Defaults to 127.0.0.1:8765 for loopback development. Any\n"
            "non-loopback bind is HTTPS-only and uses a desktop-owned\n"
            "pairing-pinned certificate. Wildcard binds require a concrete\n"
            "--advertised-host for the pairing QR.\n\n"
            "Requires vault entry REX_JWT_SECRET (at least 32 characters)."
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
    api_parser.add_argument(
        "--advertised-host",
        type=str,
        default=None,
        help="Concrete LAN host placed in pairing QR (required for wildcard binds)",
    )
    api_parser.add_argument(
        "--advertised-port",
        type=int,
        default=None,
        help="External HTTPS port placed in pairing QR (default: bind port)",
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
