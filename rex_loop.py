"""Backward-compatible launcher that simply runs :mod:`rex_assistant`."""

from rex_assistant import main


if __name__ == "__main__":  # pragma: no cover - thin wrapper
    main()
