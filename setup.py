"""Setup script to include top-level modules alongside pyproject.toml.

This setup.py is needed because pyproject.toml doesn't support py_modules
in the [tool.setuptools] section. We use this to include top-level modules
that are referenced by console scripts.
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
        "config",
        "audio_config",
        "conversation_memory",
        "flask_proxy",
    ],
)
