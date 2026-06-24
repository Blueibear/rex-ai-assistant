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

WHY data_files IS USED HERE (US-016):
  pyproject.toml [tool.setuptools] has no equivalent of data_files, and
  MANIFEST.in + include_package_data only covers files within Python package
  directories.  Bridge scripts (bridge/rex_*.py) and the config example
  (config/rex_config.example.json) live outside the rex/ package tree.

  data_files is deprecated in modern setuptools but continues to work in
  78.x.  It is the only mechanism that includes these files in the wheel
  without (a) creating a new top-level importable package or (b) duplicating
  files into rex/.  The deprecation warning is acceptable here; a future
  packaging story can revisit if setuptools removes the feature.

  After pip install, data files land at:
    {sys.prefix}/bridge/rex_*.py
    {sys.prefix}/config/rex_config.example.json
  Within the wheel zip they appear at:
    {name}-{version}.data/data/bridge/rex_*.py
    {name}-{version}.data/data/config/rex_config.example.json
"""

import glob

from setuptools import setup

setup(
    # Most configuration is in pyproject.toml.
    # This adds py_modules (unsupported in pyproject.toml) and data_files.
    py_modules=[
        "config",  # compat shim → rex.config; callers: test_llm_client, test_us013-016
        "llm_client",  # compat shim → rex.llm_client; callers: test_llm_client, test_us013-016
        "rex_speak_api",  # entry-point module; callers: wsgi.py, speak-api tests
    ],
    # US-016: include bridge scripts and config example in the wheel so that
    # `pip install .` delivers every resource documented in INSTALL.md.
    # Paths must be relative to the project root (where this file lives).
    data_files=[
        ("bridge", sorted(glob.glob("bridge/rex_*.py"))),
        ("config", ["config/rex_config.example.json"]),
    ],
)
