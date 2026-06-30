"""US-019: Console script import/help smoke tests.

One parametrized test per declared console script in pyproject.toml ensures
every entry point is importable and (for scripts with argparse) responds to
--help with a usage string.

These tests run as part of the standard pytest suite after ``pip install -e .``
and also pass on a clean wheel install.

Console scripts under test (from pyproject.toml [project.scripts]):
    rex            -> rex.cli:main
    rex-config     -> rex.config:cli
    rex-speak-api  -> rex_speak_api:main
    rex-agent      -> rex.computers.agent_server:main
    rex-gui        -> rex.gui_app:main
    rex-tool-server -> rex.openclaw.tool_server:main
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

_BASE_ENV = {
    **os.environ,
    "REX_TESTING": "true",
    "REX_FILE_LOGGING_ENABLED": "false",
}

# (script_name, module_dotpath, function_name)
# One test case per console script declared in pyproject.toml [project.scripts].
_IMPORT_CASES: list[tuple[str, str, str]] = [
    ("rex", "rex.cli", "main"),
    ("rex-config", "rex.config", "cli"),
    ("rex-speak-api", "rex_speak_api", "main"),
    ("rex-agent", "rex.computers.agent_server", "main"),
    ("rex-gui", "rex.gui_app", "main"),
    ("rex-tool-server", "rex.openclaw.tool_server", "main"),
]

# Only rex and rex-config expose argparse --help.
# Server scripts (rex-speak-api, rex-agent, rex-gui, rex-tool-server) require
# environment variables or bind to ports at startup and are unsafe to invoke
# with --help; they are covered by the import test above.
_HELP_CASES: list[tuple[str, list[str]]] = [
    ("rex", [sys.executable, "-m", "rex", "--help"]),
    (
        "rex-config",
        [
            sys.executable,
            "-c",
            ("import sys; sys.argv=['rex-config','--help'];" " from rex.config import cli; cli()"),
        ],
    ),
]


@pytest.mark.parametrize(
    "script,module,fn",
    _IMPORT_CASES,
    ids=[c[0] for c in _IMPORT_CASES],
)
def test_console_script_importable(script: str, module: str, fn: str) -> None:
    """Each declared console script entry point must be importable without error."""
    sentinel = f"{script} ok"
    code = f"from {module} import {fn}; print({sentinel!r})"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
        env=_BASE_ENV,
    )
    assert result.returncode == 0, (
        f"Import of {module}.{fn} (script: {script!r}) failed "
        f"(exit {result.returncode}):\n"
        f"stdout: {result.stdout[:500]}\n"
        f"stderr: {result.stderr[:500]}"
    )
    assert sentinel in result.stdout, (
        f"Import of {module}.{fn} did not produce expected sentinel {sentinel!r}.\n"
        f"stdout: {result.stdout[:500]}"
    )


@pytest.mark.parametrize(
    "script,cmd",
    _HELP_CASES,
    ids=[c[0] for c in _HELP_CASES],
)
def test_console_script_help(script: str, cmd: list[str]) -> None:
    """Scripts with argparse must exit 0 and print a usage string to stdout."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO_ROOT),
        env=_BASE_ENV,
    )
    assert result.returncode == 0, (
        f"'{script} --help' exited {result.returncode}:\n"
        f"stdout: {result.stdout[:500]}\n"
        f"stderr: {result.stderr[:500]}"
    )
    assert result.stdout.strip(), f"'{script} --help' produced no stdout"
    assert (
        "usage" in result.stdout.lower()
    ), f"'{script} --help' output does not contain 'usage':\n{result.stdout[:500]}"
