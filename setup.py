"""Setup script to include top-level modules alongside pyproject.toml.

WHY THIS FILE STILL EXISTS (US-259 evaluation, 2026-04-05):
  pyproject.toml does not support py_modules in [tool.setuptools], so this
  file is still needed to make backward-compatibility shims installable.

  The following shim modules have active references outside rex/ and cannot
  yet be removed. Each shim emits a DeprecationWarning at import time.
  Schedule removal once all callers have been migrated to rex.* imports:

    config.py       — tests/test_llm_client.py, tests/test_us013_openai_provider.py,
                      tests/test_us014_anthropic_provider.py,
                      tests/test_us015_local_llm_provider.py,
                      tests/test_us016_provider_routing.py
    llm_client.py   — tests/test_llm_client.py, tests/test_us013_openai_provider.py,
                      tests/test_us014_anthropic_provider.py,
                      tests/test_us015_local_llm_provider.py,
                      tests/test_us016_provider_routing.py
    rex_speak_api.py — wsgi.py, tests/test_speak_api.py, tests/test_rex_speak_api.py,
                       tests/test_us103_global_exception_handler.py,
                       tests/test_us106_graceful_shutdown.py, tests/test_us129_smoke.py

  Removed from py_modules (US-014, 2026-06-23) — files no longer exist at root
  and have no active callers outside archived/:
    rex_assistant       — no root file, no callers
    memory_utils        — no root file; archived/ callers only; active code uses rex.memory_utils
    audio_config        — no root file, no callers
    conversation_memory — no root file, no callers
"""

from setuptools import setup

setup(
    # Most configuration is in pyproject.toml.
    # This only adds the py_modules that pyproject.toml cannot specify.
    py_modules=[
        "config",        # compat shim → rex.config; callers: test_llm_client, test_us013-016
        "llm_client",    # compat shim → rex.llm_client; callers: test_llm_client, test_us013-016
        "rex_speak_api", # entry-point module; callers: wsgi.py, speak-api tests
    ],
)
