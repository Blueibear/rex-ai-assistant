"""Remote computer control (pc) commands for the Rex CLI.

Extracted verbatim from ``rex/cli.py`` (US-REM-027). Handler behavior,
argument definitions, help text, defaults, and exit codes are unchanged.
"""

from __future__ import annotations

import argparse
import sys


def _cli():
    """Return the ``rex.cli`` module at call time.

    ``rex.cli`` is the single patch point for service getters and command
    handlers (tests monkeypatch ``rex.cli.<name>``). Resolving through the
    module at call time preserves that behavior without creating an import
    cycle at module load time.
    """
    from rex import cli

    return cli


def cmd_pc(args: argparse.Namespace) -> int:
    """Manage remote Windows computers via the agent API."""
    subcommand = args.pc_command

    if subcommand == "list":
        from rex.computers.service import get_computer_service as _svc

        include_disabled = getattr(args, "all", False)
        service = _svc()
        computers = service.list_computers(include_disabled=include_disabled)

        if not computers:
            if include_disabled:
                print("No computers configured.")
            else:
                print("No enabled computers configured.")
                print("Use 'rex pc list --all' to include disabled entries.")
            return 0

        print("Configured Computers")
        print("=" * 60)
        print()

        for c in computers:
            status_tag = "[ENABLED]" if c.enabled else "[DISABLED]"
            label = f" ({c.label})" if c.label else ""
            print(f"{c.id}{label}  {status_tag}")
            print(f"  URL: {c.base_url}")
            if c.allowed_commands:
                print(f"  Allowed commands: {', '.join(c.allowed_commands)}")
            else:
                print("  Allowed commands: (none configured)")
            print()

        enabled = sum(1 for c in computers if c.enabled)
        print(f"Total: {len(computers)} computer(s), {enabled} enabled")
        return 0

    if subcommand == "status":
        from rex.computers.service import (
            ComputerDisabledError,
            ComputerNotFoundError,
            MissingTokenError,
        )
        from rex.computers.service import (
            get_computer_service as _svc,
        )

        computer_id = args.id
        service = _svc()

        try:
            result = service.status(computer_id)
        except ComputerNotFoundError as e:
            print(f"Error: {e}")
            return 1
        except ComputerDisabledError as e:
            print(f"Error: {e}")
            return 1
        except MissingTokenError as e:
            print(f"Error: {e}")
            return 1

        if not result.ok:
            print(f"Error: Could not reach computer {computer_id!r}: {result.error}")
            return 1

        print(f"Status: {computer_id}")
        print("=" * 60)
        print(f"  Hostname : {result.hostname}")
        print(f"  OS       : {result.os}")
        print(f"  User     : {result.user}")
        print(f"  Time     : {result.time}")
        return 0

    if subcommand == "run":
        from rex.computers.pc_run_policy import check_pc_run_policy
        from rex.computers.service import (
            AllowlistDeniedError,
            ComputerDisabledError,
            ComputerNotFoundError,
            MissingTokenError,
        )
        from rex.computers.service import (
            get_computer_service as _svc,
        )

        computer_id = args.id
        # args.cmd is a list from argparse nargs=argparse.REMAINDER
        cmd_parts = list(args.cmd)
        if not cmd_parts:
            print("Error: No command specified. Usage: rex pc run --id <id> -- <command> [args]")
            return 1

        command = cmd_parts[0]
        cmd_args = cmd_parts[1:] if len(cmd_parts) > 1 else []

        # ------------------------------------------------------------------
        # Policy + approvals gate (evaluated BEFORE the --yes guard so that
        # --yes cannot bypass the approval requirement).
        # ------------------------------------------------------------------

        # Resolve the active user for the approval record (best-effort).
        initiated_by: str | None = None
        try:
            from rex.identity import resolve_active_user

            initiated_by = resolve_active_user(getattr(args, "user", None))
        except Exception:  # noqa: BLE001
            pass

        # Check the client-side allowlist without a network call so the
        # approval payload can record the decision outcome.
        service = _svc()
        allowlist_matched = False
        allowlist_rule: str | None = None
        try:
            allowlist_matched, _allowed_cmds = service.get_command_allowed(computer_id, command)
            allowlist_rule = command if allowlist_matched else None
        except (ComputerNotFoundError, ComputerDisabledError) as e:
            # Let service.run() surface this error with a consistent message.
            print(f"Error: {e}")
            return 1

        # Deny non-allowlisted commands before creating any approval record.
        if not allowlist_matched:
            cfg_allowed = ", ".join(_allowed_cmds) if _allowed_cmds else "(none)"
            print(
                f"Error: Command {command!r} is not on the allowlist for"
                f" computer {computer_id!r}."
            )
            print(f"  Allowed commands: {cfg_allowed}")
            return 1

        # Consult the policy engine and approval store.
        policy_decision, approval = check_pc_run_policy(
            computer_id=computer_id,
            command=command,
            args=cmd_args,
            allowlist_matched=allowlist_matched,
            allowlist_rule=allowlist_rule,
            initiated_by=initiated_by,
        )

        if policy_decision.denied:
            print(f"Error: Remote execution denied by policy: {policy_decision.reason}")
            return 1

        if policy_decision.requires_approval:
            if approval is None or approval.status != "approved":
                # New pending approval (or existing unactioned one): block.
                if approval is not None:
                    print("Approval required before remote execution can proceed.")
                    print()
                    print(f"  Approval ID : {approval.approval_id}")
                    print(f"  Computer    : {computer_id}")
                    print(f"  Command     : {command}")
                    if cmd_args:
                        print(f"  Args        : {cmd_args}")
                    if initiated_by:
                        print(f"  Requested by: {initiated_by}")
                    print()
                    print(f"  To approve : rex approvals --approve {approval.approval_id}")
                    print(f"  To deny    : rex approvals --deny {approval.approval_id}")
                    print()
                    print("After approving, re-run this command to execute.")
                else:
                    print("Error: Approval required but could not create approval record.")
                return 1
            # approval.status == "approved": fall through to --yes guard.

        # ------------------------------------------------------------------
        # Second-layer --yes confirmation guard.
        # Reached only when policy allows auto-execute OR an approved approval
        # was found.  --yes is still required to confirm the final action.
        # ------------------------------------------------------------------
        if not getattr(args, "yes", False):
            print("Refusing to run remote command without explicit confirmation.")
            print("Remote execution is high-risk. Re-run with '--yes' if you intend to proceed.")
            return 1

        try:
            result = service.run(computer_id, command, args=cmd_args)  # type: ignore[assignment]
        except ComputerNotFoundError as e:
            print(f"Error: {e}")
            return 1
        except ComputerDisabledError as e:
            print(f"Error: {e}")
            return 1
        except MissingTokenError as e:
            print(f"Error: {e}")
            return 1
        except AllowlistDeniedError as e:
            print(f"Error: {e}")
            return 1

        if result.stdout:  # type: ignore[attr-defined]
            print(result.stdout, end="")  # type: ignore[attr-defined]
        if result.stderr:  # type: ignore[attr-defined]
            print(result.stderr, end="", file=sys.stderr)  # type: ignore[attr-defined]

        if not result.ok and result.error:
            print(f"Error: {result.error}")
            return 1

        return result.exit_code if result.exit_code >= 0 else (0 if result.ok else 1)  # type: ignore[attr-defined]

    print("Unknown pc subcommand. Use 'rex pc --help'")
    return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register this domain's subcommands on the top-level subparsers."""
    # pc (Windows computer control)
    pc_parser = subparsers.add_parser(
        "pc",
        help="Manage remote Windows computers via the agent API (client-only foundation)",
        description=(
            "Control remote Windows computers through a lightweight agent API. "
            "Requires the Rex agent server to be running on the target machine (Cycle 5.3). "
            "All commands are allowlist-checked client-side before any network call is made."
        ),
    )
    pc_subparsers = pc_parser.add_subparsers(
        title="pc commands",
        dest="pc_command",
        metavar="COMMAND",
    )

    pc_list = pc_subparsers.add_parser(
        "list",
        help="List configured computers",
        description="Show all configured remote computers and their status.",
    )
    pc_list.add_argument(
        "--all",
        action="store_true",
        help="Include disabled computers in the listing",
    )
    pc_list.set_defaults(func=_cli().cmd_pc, pc_command="list")

    pc_status = pc_subparsers.add_parser(
        "status",
        help="Check agent status on a remote computer",
        description="Query the agent API for host information (hostname, OS, user, time).",
    )
    pc_status.add_argument("--id", type=str, required=True, help="Computer ID from config")
    pc_status.set_defaults(func=_cli().cmd_pc, pc_command="status")

    pc_run = pc_subparsers.add_parser(
        "run",
        help="Run an allowlisted command on a remote computer",
        description=(
            "Execute a command on a remote computer via the agent API. "
            "The command must appear in the computer's allowlists.commands config. "
            "Use --yes to explicitly confirm high-risk remote execution. "
            "Use '--' to separate the rex options from the remote command and its arguments."
        ),
    )
    pc_run.add_argument("--id", type=str, required=True, help="Computer ID from config")
    pc_run.add_argument(
        "--yes",
        action="store_true",
        help="Confirm you want to execute a high-risk remote command",
    )
    pc_run.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command and arguments to run on the remote computer (e.g. -- whoami)",
    )
    pc_run.set_defaults(func=_cli().cmd_pc, pc_command="run")

    pc_parser.set_defaults(func=_cli().cmd_pc, pc_command="list")
