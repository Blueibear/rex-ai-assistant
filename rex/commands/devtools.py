"""Developer tools (browser, os, gh, code) commands for the Rex CLI.

Extracted verbatim from ``rex/cli.py`` (US-REM-027). Handler behavior,
argument definitions, help text, defaults, and exit codes are unchanged.
"""

from __future__ import annotations

import argparse
import sys
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


def cmd_browser(args: argparse.Namespace) -> int:
    """Manage browser automation."""
    import asyncio
    import json

    subcommand = args.browser_command

    if subcommand == "run":
        script_path = Path(args.script)
        if not script_path.exists():
            print(f"Error: Script file not found: {script_path}")
            return 1

        try:
            with open(script_path, encoding="utf-8") as f:
                script_data = json.load(f)

            steps = script_data.get("steps", [])
            headless = not args.headed

            print(f"Running browser script: {script_path}")
            print(f"  Steps: {len(steps)}")
            print(f"  Headless: {headless}")
            print()

            bridge = _cli().get_browser_service()
            results = asyncio.run(bridge.execute_script(str(script_path), headless=headless))

            for result in results:
                step_num = result.get("step", "?")
                action = result.get("action", "unknown")
                status = result.get("status", "unknown")

                if status == "success":
                    print(f"✓ Step {step_num}: {action}")
                elif status == "error":
                    print(f"✗ Step {step_num}: {action} - {result.get('error', 'Unknown error')}")
                else:
                    print(f"⊘ Step {step_num}: {action} - {status}")

            print()
            success_count = sum(1 for r in results if r.get("status") == "success")
            print(f"Completed: {success_count}/{len(results)} steps successful")

            return 0 if success_count == len(results) else 1

        except Exception as e:
            print(f"Error: {e}")
            return 1

    if subcommand == "sessions":
        service = _cli().get_browser_service()
        sessions = service.list_sessions()

        if not sessions:
            print("No browser sessions found.")
            return 0

        print("Browser Sessions")
        print("=" * 60)
        print()

        for session in sessions:
            print(f"  {session}")

        print()
        print(f"Total: {len(sessions)} sessions")
        return 0

    if subcommand == "screenshots":
        service = _cli().get_browser_service()
        screenshots = service.list_screenshots()

        if not screenshots:
            print("No screenshots found.")
            return 0

        print("Screenshots")
        print("=" * 60)
        print()

        for screenshot in screenshots:
            print(f"  {screenshot}")

        print()
        print(f"Total: {len(screenshots)} screenshots")
        return 0

    print("Unknown browser subcommand. Use 'rex browser --help'")
    return 1


def cmd_os(args: argparse.Namespace) -> int:
    """Manage OS automation."""
    subcommand = args.os_command
    service = _cli().get_os_service()

    if subcommand == "run":
        command = args.command.split()

        try:
            print(f"Running command: {' '.join(command)}")
            print()

            result = service.run_command(command)

            if result.stdout:
                print(result.stdout)

            if result.stderr:
                print(f"[stderr] {result.stderr}", file=sys.stderr)

            print()
            print(f"Exit code: {result.returncode}")
            print(f"Duration: {result.duration_ms}ms")

            return result.returncode  # type: ignore[no-any-return]

        except PermissionError as e:
            print(f"Error: {e}")
            return 1
        except Exception as e:
            print(f"Error: {e}")
            return 1

    if subcommand == "copy":
        try:
            result = service.copy_file(args.src, args.dst)
            print(f"Copied: {result['src']} -> {result['dst']}")
            print(f"Size: {result['size']} bytes")
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1

    if subcommand == "move":
        try:
            result = service.move_file(args.src, args.dst)
            print(f"Moved: {result['src']} -> {result['dst']}")
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1

    if subcommand == "delete":
        try:
            backup = not args.permanent
            result = service.delete_file(args.path, backup=backup)

            if result["action"] == "moved_to_trash":
                print(f"Moved to trash: {result['path']}")
                print(f"Backup location: {result['backup_path']}")
            else:
                print(f"Permanently deleted: {result['path']}")

            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1

    if subcommand == "trash":
        files = service.list_trash()

        if not files:
            print("Trash is empty.")
            return 0

        print("Trash Contents")
        print("=" * 80)
        print()

        for file in files:
            print(f"{file['name']}")
            print(f"  Path: {file['path']}")
            print(f"  Size: {file['size']} bytes")
            print(f"  Modified: {file['modified']}")
            print()

        print(f"Total: {len(files)} files")
        return 0

    print("Unknown os subcommand. Use 'rex os --help'")
    return 1


def cmd_gh(args: argparse.Namespace) -> int:
    """Manage GitHub integration."""
    subcommand = args.gh_command
    service = _cli().get_github_service()

    if subcommand == "repos":
        try:
            repos = service.list_repos()

            if not repos:
                print("No repositories found.")
                return 0

            print("GitHub Repositories")
            print("=" * 80)
            print()

            for repo in repos:
                visibility = "private" if repo.private else "public"
                print(f"{repo.full_name} ({visibility})")
                if repo.description:
                    print(f"  {repo.description}")
                print(f"  URL: {repo.url}")
                print(f"  Default branch: {repo.default_branch}")
                print()

            print(f"Total: {len(repos)} repositories")
            return 0

        except Exception as e:
            print(f"Error: {e}")
            return 1

    if subcommand == "prs":
        try:
            prs = service.list_prs(args.repo, state=args.state)

            if not prs:
                print(f"No pull requests found ({args.state}).")
                return 0

            print(f"Pull Requests for {args.repo}")
            print("=" * 80)
            print()

            for pr in prs:
                print(f"#{pr.number}: {pr.title}")
                print(f"  State: {pr.state}")
                print(f"  Author: {pr.author}")
                print(f"  Branch: {pr.head_branch} -> {pr.base_branch}")
                print(f"  URL: {pr.url}")
                print(f"  Created: {pr.created_at}")
                print()

            print(f"Total: {len(prs)} pull requests")
            return 0

        except Exception as e:
            print(f"Error: {e}")
            return 1

    if subcommand == "issue-create":
        try:
            labels = args.labels.split(",") if args.labels else None
            issue = service.create_issue(args.repo, args.title, args.body, labels)

            print("Issue created successfully!")
            print(f"  Number: #{issue.number}")
            print(f"  Title: {issue.title}")
            print(f"  URL: {issue.url}")

            return 0

        except Exception as e:
            print(f"Error: {e}")
            return 1

    if subcommand == "pr-create":
        try:
            pr = service.create_pr(
                args.repo,
                args.head,
                args.base,
                args.title,
                args.body,
            )

            print("Pull request created successfully!")
            print(f"  Number: #{pr.number}")
            print(f"  Title: {pr.title}")
            print(f"  URL: {pr.url}")

            return 0

        except Exception as e:
            print(f"Error: {e}")
            return 1

    print("Unknown gh subcommand. Use 'rex gh --help'")
    return 1


def cmd_code(args: argparse.Namespace) -> int:
    """Manage VS Code operations."""
    subcommand = args.code_command
    service = _cli().get_vscode_service()

    if subcommand == "open":
        try:
            result = service.open_file(args.path)

            print(f"File: {result['path']}")
            print(f"Lines: {result['lines']}")
            print(f"Size: {result['size']} bytes")
            print(f"Modified: {result['modified']}")
            print()
            print("-" * 80)
            print(result["content"])
            print("-" * 80)

            return 0

        except Exception as e:
            print(f"Error: {e}")
            return 1

    if subcommand == "patch":
        try:
            from pathlib import Path

            patch_file = Path(args.patch_file)
            if not patch_file.exists():
                print(f"Error: Patch file not found: {patch_file}")
                return 1

            with open(patch_file, encoding="utf-8") as f:
                patch_content = f.read()

            result = service.apply_patch(args.path, patch_content)

            if result.success:
                print("Patch applied successfully!")
                print(f"  File: {result.file_path}")
                print(f"  Hunks applied: {result.hunks_applied}")
            else:
                print("Patch application failed!")
                print(f"  File: {result.file_path}")
                print(f"  Hunks applied: {result.hunks_applied}")
                print(f"  Hunks failed: {result.hunks_failed}")
                if result.errors:
                    print("  Errors:")
                    for error in result.errors:
                        print(f"    - {error}")
                return 1

            return 0

        except Exception as e:
            print(f"Error: {e}")
            return 1

    if subcommand == "test":
        try:
            result = service.run_tests(
                test_path=args.path,
                pattern=args.pattern,
                verbose=args.verbose,
            )

            print("Test Results")
            print("=" * 80)
            print()

            if result.success:
                print("✓ All tests passed!")
            else:
                print("✗ Some tests failed")

            print()
            print(f"Total: {result.total}")
            print(f"Passed: {result.passed}")
            print(f"Failed: {result.failed}")
            print(f"Errors: {result.errors}")
            print(f"Skipped: {result.skipped}")
            print(f"Duration: {result.duration_seconds:.2f}s")

            if args.verbose:
                print()
                print("Output:")
                print("-" * 80)
                print(result.output)

            return 0 if result.success else 1

        except Exception as e:
            print(f"Error: {e}")
            return 1

    print("Unknown code subcommand. Use 'rex code --help'")
    return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register this domain's subcommands on the top-level subparsers."""
    # browser
    browser_parser = subparsers.add_parser(
        "browser",
        help="Browser automation with Playwright",
        description="Automate browser interactions with Playwright.",
    )
    browser_subparsers = browser_parser.add_subparsers(
        title="browser commands",
        dest="browser_command",
        metavar="COMMAND",
    )

    browser_run = browser_subparsers.add_parser(
        "run",
        help="Run a browser automation script",
        description="Execute a browser automation script from a JSON file.",
    )
    browser_run.add_argument("script", type=str, help="Path to the script JSON file")
    browser_run.add_argument(
        "--headed", action="store_true", help="Run in headed mode (show browser)"
    )
    browser_run.set_defaults(func=_cli().cmd_browser, browser_command="run")

    browser_sessions = browser_subparsers.add_parser(
        "sessions",
        help="List browser sessions",
        description="List available browser sessions.",
    )
    browser_sessions.set_defaults(func=_cli().cmd_browser, browser_command="sessions")

    browser_screenshots = browser_subparsers.add_parser(
        "screenshots",
        help="List captured screenshots",
        description="List all captured screenshots.",
    )
    browser_screenshots.set_defaults(func=_cli().cmd_browser, browser_command="screenshots")

    browser_parser.set_defaults(func=_cli().cmd_browser, browser_command="sessions")

    # os
    os_parser = subparsers.add_parser(
        "os",
        help="OS automation (safe command execution)",
        description="Execute system commands and file operations with safety rails.",
    )
    os_subparsers = os_parser.add_subparsers(
        title="os commands",
        dest="os_command",
        metavar="COMMAND",
    )

    os_run = os_subparsers.add_parser(
        "run",
        help="Run a safe system command",
        description="Execute a whitelisted system command.",
    )
    os_run.add_argument("command", type=str, help="Command to run (e.g., 'ls -la')")
    os_run.set_defaults(func=_cli().cmd_os, os_command="run")

    os_copy = os_subparsers.add_parser(
        "copy",
        help="Copy a file",
        description="Copy a file with path sanitization.",
    )
    os_copy.add_argument("src", type=str, help="Source file path")
    os_copy.add_argument("dst", type=str, help="Destination file path")
    os_copy.set_defaults(func=_cli().cmd_os, os_command="copy")

    os_move = os_subparsers.add_parser(
        "move",
        help="Move a file",
        description="Move a file with path sanitization.",
    )
    os_move.add_argument("src", type=str, help="Source file path")
    os_move.add_argument("dst", type=str, help="Destination file path")
    os_move.set_defaults(func=_cli().cmd_os, os_command="move")

    os_delete = os_subparsers.add_parser(
        "delete",
        help="Delete a file",
        description="Delete a file (moves to trash by default).",
    )
    os_delete.add_argument("path", type=str, help="File path to delete")
    os_delete.add_argument(
        "--permanent", action="store_true", help="Permanently delete (skip trash)"
    )
    os_delete.set_defaults(func=_cli().cmd_os, os_command="delete")

    os_trash = os_subparsers.add_parser(
        "trash",
        help="List trash contents",
        description="Show files in the trash folder.",
    )
    os_trash.set_defaults(func=_cli().cmd_os, os_command="trash")

    os_parser.set_defaults(func=_cli().cmd_os, os_command="trash")

    # gh (GitHub)
    gh_parser = subparsers.add_parser(
        "gh",
        help="GitHub integration",
        description="Manage GitHub repositories, issues, and pull requests.",
    )
    gh_subparsers = gh_parser.add_subparsers(
        title="github commands",
        dest="gh_command",
        metavar="COMMAND",
    )

    gh_repos = gh_subparsers.add_parser(
        "repos",
        help="List repositories",
        description="List repositories accessible to the authenticated user.",
    )
    gh_repos.set_defaults(func=_cli().cmd_gh, gh_command="repos")

    gh_prs = gh_subparsers.add_parser(
        "prs",
        help="List pull requests",
        description="List pull requests for a repository.",
    )
    gh_prs.add_argument("repo", type=str, help="Repository in format 'owner/repo'")
    gh_prs.add_argument(
        "--state",
        type=str,
        default="open",
        choices=["open", "closed", "all"],
        help="PR state filter",
    )
    gh_prs.set_defaults(func=_cli().cmd_gh, gh_command="prs")

    gh_issue_create = gh_subparsers.add_parser(
        "issue-create",
        help="Create an issue",
        description="Create a new issue in a repository.",
    )
    gh_issue_create.add_argument("repo", type=str, help="Repository in format 'owner/repo'")
    gh_issue_create.add_argument("--title", type=str, required=True, help="Issue title")
    gh_issue_create.add_argument("--body", type=str, required=True, help="Issue body")
    gh_issue_create.add_argument("--labels", type=str, help="Comma-separated list of labels")
    gh_issue_create.set_defaults(func=_cli().cmd_gh, gh_command="issue-create")

    gh_pr_create = gh_subparsers.add_parser(
        "pr-create",
        help="Create a pull request",
        description="Create a new pull request.",
    )
    gh_pr_create.add_argument("repo", type=str, help="Repository in format 'owner/repo'")
    gh_pr_create.add_argument("--head", type=str, required=True, help="Head branch")
    gh_pr_create.add_argument("--base", type=str, required=True, help="Base branch")
    gh_pr_create.add_argument("--title", type=str, required=True, help="PR title")
    gh_pr_create.add_argument("--body", type=str, required=True, help="PR body")
    gh_pr_create.set_defaults(func=_cli().cmd_gh, gh_command="pr-create")

    gh_parser.set_defaults(func=_cli().cmd_gh, gh_command="repos")

    # code (VS Code integration)
    code_parser = subparsers.add_parser(
        "code",
        help="VS Code integration",
        description="File operations, patch application, and test execution.",
    )
    code_subparsers = code_parser.add_subparsers(
        title="code commands",
        dest="code_command",
        metavar="COMMAND",
    )

    code_open = code_subparsers.add_parser(
        "open",
        help="Open and display a file",
        description="Read and display file contents.",
    )
    code_open.add_argument("path", type=str, help="File path to open")
    code_open.set_defaults(func=_cli().cmd_code, code_command="open")

    code_patch = code_subparsers.add_parser(
        "patch",
        help="Apply a patch to a file",
        description="Apply a unified diff patch to a file.",
    )
    code_patch.add_argument("path", type=str, help="File path to patch")
    code_patch.add_argument("--patch-file", type=str, required=True, help="Path to patch/diff file")
    code_patch.set_defaults(func=_cli().cmd_code, code_command="patch")

    code_test = code_subparsers.add_parser(
        "test",
        help="Run tests",
        description="Execute tests using pytest.",
    )
    code_test.add_argument("--path", type=str, help="Test file or directory (default: tests/)")
    code_test.add_argument("--pattern", type=str, help="Test pattern filter (e.g., 'test_browser')")
    code_test.add_argument("-v", "--verbose", action="store_true", help="Verbose test output")
    code_test.set_defaults(func=_cli().cmd_code, code_command="test")

    code_parser.set_defaults(func=_cli().cmd_code, code_command="open")
