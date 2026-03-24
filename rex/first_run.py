"""First-run detection and guided setup message for Rex.

Prints a short onboarding message when Rex is started for the first time
(i.e., no config file exists or the config file is empty / contains only
an empty JSON object).  On subsequent runs the message is suppressed.
"""

from __future__ import annotations

import json
from pathlib import Path

_DEFAULT_CONFIG_PATH = Path("config/rex_config.json")

_WELCOME_MESSAGE = """\
============================================================
Welcome to Rex. Let's get you set up.
============================================================

Here are the 3 most important next steps:

  1. Configure your LLM provider
     Edit config/rex_config.json and set the 'openai.base_url'
     to your LM Studio or compatible API endpoint, e.g.:
       "openai": {"base_url": "http://127.0.0.1:1234/v1"}

  2. Run the environment health check
     Execute:  rex doctor
     This verifies your audio devices, model paths, and
     network connections are working correctly.

  3. Start chatting
     Execute:  rex
     Rex will start in text-chat mode once your LLM is ready.

For full documentation see: docs/usage.md
============================================================
"""


def is_first_run(config_path: str | Path = _DEFAULT_CONFIG_PATH) -> bool:
    """Return True when *config_path* does not exist or is effectively empty.

    "Effectively empty" means the file contains the empty JSON object ``{}``
    or only whitespace / zero bytes.
    """
    path = Path(config_path)
    if not path.exists():
        return True
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return True
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Unreadable / corrupt config — treat as first run so the user notices.
        return True
    if isinstance(data, dict) and len(data) == 0:
        return True
    return False


def print_welcome_message() -> None:
    """Print the first-run guided setup message to stdout."""
    print(_WELCOME_MESSAGE)


def maybe_print_welcome(
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
) -> bool:
    """Print the welcome message if this looks like a first run.

    Returns True if the message was printed, False otherwise.
    """
    if is_first_run(config_path):
        print_welcome_message()
        return True
    return False


__all__ = [
    "is_first_run",
    "print_welcome_message",
    "maybe_print_welcome",
]
