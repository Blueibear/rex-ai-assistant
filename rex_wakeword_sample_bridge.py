"""Compatibility entrypoint for ``bridge/rex_wakeword_sample_bridge.py``.

This root-level module keeps legacy Electron/test imports working while the
maintained bridge implementation lives under ``bridge/``.  Executing the
canonical source in this module namespace preserves patchable helpers such as
``_handle_list`` and ``_CONFIG_DIR_DEFAULT`` for existing tests.
"""

from __future__ import annotations

from pathlib import Path

_BRIDGE_PATH = Path(__file__).resolve().parent / "bridge" / "rex_wakeword_sample_bridge.py"
exec(compile(_BRIDGE_PATH.read_text(encoding="utf-8"), str(_BRIDGE_PATH), "exec"), globals())
