"""Usage, shopping, history, and quick actions commands for the Rex CLI.

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


def cmd_usage(args: argparse.Namespace) -> int:
    """Print LLM usage summary."""
    from rex.llm_usage import summarise

    summary = summarise()
    total_requests = summary["total_requests"]
    total_tokens = summary["total_tokens"]
    by_model = summary["by_model"]

    if total_requests == 0:
        print("No LLM usage recorded yet.")
        return 0

    print("LLM Usage Summary")
    print("=" * 50)
    print(f"Total requests : {total_requests}")
    print(f"Total tokens   : {total_tokens}")
    print()
    print("By model:")
    for model, stats in sorted(by_model.items()):
        print(f"  {model}")
        print(f"    Requests          : {stats['requests']}")
        print(f"    Prompt tokens     : {stats['prompt_tokens']}")
        print(f"    Completion tokens : {stats['completion_tokens']}")
        print(f"    Total tokens      : {stats['total_tokens']}")
    return 0


def cmd_shopping(args: argparse.Namespace) -> int:
    """Manage the shopping list via CLI."""
    from rex.shopping_list import ShoppingList

    sl = ShoppingList()
    command = getattr(args, "shopping_command", "list")

    if command == "list":
        items = sl.list_items()
        if not items:
            print("Shopping list is empty.")
            return 0
        unchecked = [i for i in items if not i.checked]
        checked = [i for i in items if i.checked]
        if unchecked:
            print("[ ] Items to buy:")
            for item in unchecked:
                qty = f" x{item.quantity}" if item.quantity != 1.0 else ""
                unit = f" {item.unit}" if item.unit else ""
                print(f"  - {item.name}{qty}{unit}")
        if checked:
            print("[✓] Checked:")
            for item in checked:
                print(f"  - {item.name}")
        return 0

    if command == "add":
        name = " ".join(args.name) if isinstance(args.name, list) else args.name
        name = name.strip()
        if not name:
            print("Error: item name is required.")
            return 1
        item = sl.add_item(
            name,
            quantity=getattr(args, "quantity", 1.0),
            unit=getattr(args, "unit", ""),
            added_by="cli",
        )
        print(f"Added: {item.name}")
        return 0

    if command == "clear":
        sl.clear_checked()
        print("Cleared checked items.")
        return 0

    print(f"Unknown shopping command: {command}")
    return 1


def cmd_history(args: argparse.Namespace) -> int:
    """Show recent command history."""
    from rex.command_history import CommandHistoryStore

    limit = getattr(args, "limit", 20)
    store = CommandHistoryStore()
    entries = store.get_recent(limit=limit)
    if not entries:
        print("No command history.")
        return 0
    print(f"Recent {len(entries)} command(s):")
    print("-" * 60)
    for entry in entries:
        ts = entry.get("timestamp", "")
        cmd = entry.get("command", "")
        ok = "✓" if entry.get("success", True) else "✗"
        result = entry.get("result", "")
        result_preview = (result[:60] + "…") if len(result) > 60 else result
        print(f"{ok} [{ts}] {cmd}")
        if result_preview:
            print(f"   → {result_preview}")
    return 0


def cmd_quick_actions(args: argparse.Namespace) -> int:
    """List quick actions from a user's Memory profile."""
    from rex.identity import get_user_profile, list_known_users

    user_arg = getattr(args, "user", None)
    command = getattr(args, "qa_command", "list")

    if command == "list":
        if user_arg:
            user_ids = [user_arg]
        else:
            user_ids = [u["id"] for u in (list_known_users() or [])]
            if not user_ids:
                print("No users found.")
                return 0

        for uid in user_ids:
            profile = get_user_profile(uid) or {}
            prefs = profile.get("preferences", {})
            actions = prefs.get("quick_actions", [])
            if isinstance(actions, list) and actions:
                print(f"Quick actions for {uid}:")
                for action in actions:
                    print(
                        f"  [{action.get('id', '?')}] {action.get('label', '')} → {action.get('command', '')}"
                    )
            else:
                print(f"No quick actions for {uid}.")
        return 0

    print(f"Unknown quick-actions command: {command}")
    return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register this domain's subcommands on the top-level subparsers."""
    # usage
    usage_parser = subparsers.add_parser(
        "usage",
        help="Show LLM usage summary (total requests, tokens, by model)",
        description="Display per-model token usage recorded from Ollama and other backends.",
    )
    usage_parser.set_defaults(func=_cli().cmd_usage)

    # shopping
    shopping_parser = subparsers.add_parser(
        "shopping",
        help="Manage the shopping list",
        description="View, add, and clear items on the shopping list.",
    )
    shopping_subparsers = shopping_parser.add_subparsers(
        title="shopping commands",
        dest="shopping_command",
        metavar="COMMAND",
    )
    shopping_list_cmd = shopping_subparsers.add_parser(
        "list",
        help="Show all shopping list items",
    )
    shopping_list_cmd.set_defaults(func=_cli().cmd_shopping, shopping_command="list")

    shopping_add_cmd = shopping_subparsers.add_parser(
        "add",
        help="Add an item to the shopping list",
    )
    shopping_add_cmd.add_argument("name", nargs="+", help="Item name")
    shopping_add_cmd.add_argument(
        "--quantity", type=float, default=1.0, help="Quantity (default: 1)"
    )
    shopping_add_cmd.add_argument("--unit", type=str, default="", help="Unit (e.g. kg, pcs)")
    shopping_add_cmd.set_defaults(func=_cli().cmd_shopping, shopping_command="add")

    shopping_clear_cmd = shopping_subparsers.add_parser(
        "clear",
        help="Clear checked items from the shopping list",
    )
    shopping_clear_cmd.set_defaults(func=_cli().cmd_shopping, shopping_command="clear")

    shopping_parser.set_defaults(func=_cli().cmd_shopping, shopping_command="list")

    # history
    history_parser = subparsers.add_parser(
        "history",
        help="Show recent command history",
        description="Display the most recent commands sent to Rex and their outcomes.",
    )
    history_parser.add_argument(
        "--limit", type=int, default=20, help="Number of entries to show (default: 20)"
    )
    history_parser.set_defaults(func=_cli().cmd_history)

    # quick-actions
    qa_parser = subparsers.add_parser(
        "quick-actions",
        help="List quick actions from user Memory profiles",
        description="Show one-click quick actions stored per user.",
    )
    qa_parser.add_argument(
        "--user", type=str, default=None, help="User ID to filter by (default: all users)"
    )
    qa_subparsers = qa_parser.add_subparsers(
        title="quick-actions commands",
        dest="qa_command",
        metavar="COMMAND",
    )
    qa_list_cmd = qa_subparsers.add_parser("list", help="List quick actions")
    qa_list_cmd.add_argument(
        "--user", type=str, default=None, help="User ID to filter by (default: all users)"
    )
    qa_list_cmd.set_defaults(func=_cli().cmd_quick_actions, qa_command="list")
    qa_parser.set_defaults(func=_cli().cmd_quick_actions, qa_command="list")
