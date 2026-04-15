"""Regression tests for US-253: wakeword root-file hygiene.

Verifies that root-level wakeword_utils.py and wakeword_listener.py have been
removed and that the canonical implementation in rex/wakeword/ is intact.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_root_wakeword_utils_removed():
    assert not (
        REPO_ROOT / "wakeword_utils.py"
    ).exists(), "wakeword_utils.py must not exist at repo root; use rex.wakeword_utils"


def test_root_wakeword_listener_removed():
    assert not (
        REPO_ROOT / "wakeword_listener.py"
    ).exists(), "wakeword_listener.py must not exist at repo root; use rex.wakeword.listener"


def test_canonical_wakeword_utils_importable():
    from rex.wakeword_utils import detect_wakeword, load_wakeword_model  # noqa: F401

    assert callable(detect_wakeword)
    assert callable(load_wakeword_model)


def test_canonical_wakeword_package_importable():
    from rex.wakeword.utils import detect_wakeword, load_wakeword_model  # noqa: F401

    assert callable(detect_wakeword)
    assert callable(load_wakeword_model)
