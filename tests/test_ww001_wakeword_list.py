"""Tests for US-WW-001: Wake word selection via UI — bridge and listing."""

from __future__ import annotations

import contextlib
import io
import json
import sys
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# rex_wakeword_list_bridge — black-box via in-process invocation
# ---------------------------------------------------------------------------


def _run_bridge(stdin_data: str = "{}") -> dict:
    """Run the wake word list bridge in-process and return parsed JSON output."""
    captured = io.StringIO()
    original_stdin = sys.stdin
    sys.stdin = io.StringIO(stdin_data)
    try:
        with contextlib.redirect_stdout(captured):
            import rex_wakeword_list_bridge  # noqa: PLC0415

            try:
                rex_wakeword_list_bridge.main()
            except SystemExit:
                pass
    finally:
        sys.stdin = original_stdin
    return json.loads(captured.getvalue().strip())


def test_bridge_returns_ok() -> None:
    """Bridge always returns ok=True with a non-empty wake_words list."""
    result = _run_bridge()
    assert result["ok"] is True
    assert isinstance(result["wake_words"], list)
    assert len(result["wake_words"]) > 0


def test_bridge_wake_words_have_required_fields() -> None:
    """Each wake word entry has id, name, and engine fields."""
    result = _run_bridge()
    for ww in result["wake_words"]:
        assert "id" in ww
        assert "name" in ww
        assert "engine" in ww
        assert isinstance(ww["id"], str)
        assert isinstance(ww["name"], str)
        assert isinstance(ww["engine"], str)


def test_bridge_engine_is_openwakeword() -> None:
    """All entries report engine == 'openwakeword'."""
    result = _run_bridge()
    for ww in result["wake_words"]:
        assert ww["engine"] == "openwakeword"


def test_bridge_falls_back_to_defaults_when_oww_unavailable() -> None:
    """Bridge returns default keyword list when openWakeWord is not installed."""
    with patch.dict("sys.modules", {"openwakeword": None}):
        result = _run_bridge()
    assert result["ok"] is True
    assert len(result["wake_words"]) >= 5  # at least the 5 default keywords


def test_bridge_uses_openwakeword_models_when_available() -> None:
    """Bridge uses openwakeword.MODELS when the module is available."""
    mock_oww = MagicMock()
    mock_oww.MODELS = {"hey_test": {}, "ok_test": {}}
    with patch.dict("sys.modules", {"openwakeword": mock_oww}):
        result = _run_bridge()
    assert result["ok"] is True
    ids = {ww["id"] for ww in result["wake_words"]}
    assert "hey_test" in ids or "ok_test" in ids


def test_bridge_id_has_no_spaces() -> None:
    """Wake word ids use underscores, not spaces."""
    result = _run_bridge()
    for ww in result["wake_words"]:
        assert " " not in ww["id"], f"id {ww['id']!r} should not contain spaces"


# ---------------------------------------------------------------------------
# list_openwakeword_keywords helper
# ---------------------------------------------------------------------------


def test_list_openwakeword_keywords_from_dict() -> None:
    from rex.wakeword.selection import list_openwakeword_keywords

    mock_module = MagicMock()
    mock_module.MODELS = {"hey_jarvis": {}, "hey_mycroft": {}}
    result = list_openwakeword_keywords(mock_module)
    assert "hey jarvis" in result
    assert "hey mycroft" in result


def test_list_openwakeword_keywords_none_module() -> None:
    from rex.wakeword.selection import list_openwakeword_keywords

    result = list_openwakeword_keywords(None)
    assert result == []


def test_list_openwakeword_keywords_no_models_attr() -> None:
    from rex.wakeword.selection import list_openwakeword_keywords

    mock_module = MagicMock(spec=[])  # no MODELS attribute
    result = list_openwakeword_keywords(mock_module)
    assert result == []
