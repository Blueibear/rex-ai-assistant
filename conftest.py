"""Project-wide pytest configuration."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence


def _has_pytest_cov() -> bool:
    return importlib.util.find_spec("pytest_cov") is not None


def _has_option(args: Sequence[str], option: str) -> bool:
    option_with_equals = f"{option}="
    for arg in args:
        if arg == option or arg.startswith(option_with_equals):
            return True
    return False


def pytest_load_initial_conftests(early_config, parser, args) -> None:
    """Conditionally inject pytest-cov options when the plugin is available."""
    if not _has_pytest_cov():
        return

    if not _has_option(args, "--cov"):
        args.append("--cov=rex")
    if not _has_option(args, "--cov-report"):
        args.append("--cov-report=term-missing")
    if not _has_option(args, "--cov-fail-under"):
        args.append("--cov-fail-under=25")
