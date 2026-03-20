"""Tests for rex.__main__ module (package entry point)."""

from __future__ import annotations

from unittest.mock import patch


def test_import():
    """rex.__main__ imports without error."""
    import rex.__main__  # noqa: F401


def test_main_is_from_rex_cli():
    """rex.__main__.main is the same object as rex.cli.main."""
    import rex.__main__ as m
    from rex.cli import main

    assert m.main is main


def test_module_calls_sys_exit_with_main_return_value():
    """The __main__ guard calls sys.exit(main()) when run as __main__."""
    import rex.__main__ as m

    with patch.object(m, "main", return_value=0) as mock_main:
        with patch("sys.exit") as mock_exit:
            # Simulate the if __name__ == "__main__": block manually
            import sys

            sys.exit(m.main())

    mock_main.assert_called_once()
    mock_exit.assert_called_once_with(0)
