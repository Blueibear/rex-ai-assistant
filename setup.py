"""Setup script to include top-level modules alongside pyproject.toml.

WHY THIS FILE STILL EXISTS (US-259 evaluation, 2026-04-05):
  pyproject.toml does not support py_modules in [tool.setuptools], so this
  file is still needed to make backward-compatibility shims installable.

  The following shim modules have active references outside rex/ and cannot
  yet be removed. Each shim now emits a DeprecationWarning at import time.
  Schedule removal once all callers have been migrated to rex.* imports:

    config.py       — gui.py, tests/test_llm_client.py, tests/test_memory_utils.py,
                      tests/test_us013-016_*.py
    llm_client.py   — tests/test_llm_client.py, tests/test_us013-016_*.py
    memory_utils.py — flask_proxy.py, gui.py, tests/test_memory_utils.py
    logging_utils.py— audio_config.py, gui.py, gui_settings_tab.py, install.py

  The following shims have no known callers but are kept for safety until the
  next cleanup cycle: audio_config.py, conversation_memory.py, flask_proxy.py,
  python_compat.py, rex_assistant.py.
"""

from setuptools import setup

setup(
    # Most configuration is in pyproject.toml
    # This only adds the py_modules that can't be specified there
    py_modules=[
        "rex_assistant",
        "rex_speak_api",
        "llm_client",
        "memory_utils",
        "python_compat",
        "config",
        "audio_config",
        "conversation_memory",
        "flask_proxy",
    ],
)
