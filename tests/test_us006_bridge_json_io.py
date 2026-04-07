"""US-006: Smoke tests for rex_shopping_list_bridge and rex_speaker_bridge.

Verifies:
- Each bridge accepts JSON on stdin and returns valid JSON on stdout
- Each bridge returns {"error": "..."} on invalid (non-JSON) input
- Each bridge returns {"error": "..."} on unknown command
"""
from __future__ import annotations

import json
from io import StringIO
from unittest.mock import MagicMock, patch


def _run_main(module_name: str, stdin_data: str) -> dict:
    """Import a bridge module and invoke main() with the given stdin, returning parsed output."""
    import importlib

    mod = importlib.import_module(module_name)

    captured = StringIO()
    with patch("sys.stdin", StringIO(stdin_data)):
        with patch("sys.stdout", captured):
            try:
                mod.main()
            except SystemExit:
                pass

    output = captured.getvalue().strip()
    assert output, f"{module_name}: no output produced"
    return json.loads(output)


# ---------------------------------------------------------------------------
# rex_shopping_list_bridge
# ---------------------------------------------------------------------------

SHOPPING_MODULE = "rex_shopping_list_bridge"


def _mock_shopping_item(name: str = "eggs") -> MagicMock:
    item = MagicMock()
    item.to_dict.return_value = {"id": "abc", "name": name, "quantity": 1.0, "unit": "", "checked": False}
    return item


class TestShoppingListBridgeJsonIO:
    def _with_mock_sl(self, items=None):
        mock_sl = MagicMock()
        mock_sl.list_items.return_value = items or []
        return patch("rex.shopping_list.ShoppingList", return_value=mock_sl)

    def test_list_returns_valid_json(self):
        mock_sl = MagicMock()
        mock_sl.list_items.return_value = []
        with patch("rex.shopping_list.ShoppingList", return_value=mock_sl):
            result = _run_main(SHOPPING_MODULE, '{"command": "list"}')
        assert isinstance(result, dict)
        assert "items" in result or "error" in result

    def test_action_list_returns_valid_json(self):
        mock_sl = MagicMock()
        mock_sl.list_items.return_value = []
        with patch("rex.shopping_list.ShoppingList", return_value=mock_sl):
            result = _run_main(SHOPPING_MODULE, '{"action": "list"}')
        assert isinstance(result, dict)
        assert "items" in result or "error" in result

    def test_invalid_input_returns_error_key(self):
        result = _run_main(SHOPPING_MODULE, "notjson{{{{")
        assert "error" in result

    def test_unknown_command_returns_error_key(self):
        mock_sl = MagicMock()
        with patch("rex.shopping_list.ShoppingList", return_value=mock_sl):
            result = _run_main(SHOPPING_MODULE, '{"command": "bogus_cmd"}')
        assert "error" in result


# ---------------------------------------------------------------------------
# rex_speaker_bridge
# ---------------------------------------------------------------------------

SPEAKER_MODULE = "rex_speaker_bridge"


class TestSpeakerBridgeJsonIO:
    def test_list_returns_valid_json(self):
        mock_svc = MagicMock()
        mock_svc.discover_now.return_value = []
        with patch.dict(
            "sys.modules",
            {
                "rex.audio.speaker_discovery": MagicMock(
                    SpeakerDiscoveryService=MagicMock(return_value=mock_svc)
                )
            },
        ):
            result = _run_main(SPEAKER_MODULE, '{"command": "list"}')
        assert isinstance(result, dict)
        assert "speakers" in result or "error" in result

    def test_action_list_returns_valid_json(self):
        mock_svc = MagicMock()
        mock_svc.discover_now.return_value = []
        with patch.dict(
            "sys.modules",
            {
                "rex.audio.speaker_discovery": MagicMock(
                    SpeakerDiscoveryService=MagicMock(return_value=mock_svc)
                )
            },
        ):
            result = _run_main(SPEAKER_MODULE, '{"action": "list"}')
        assert isinstance(result, dict)
        assert "speakers" in result or "error" in result

    def test_invalid_input_returns_error_key(self):
        result = _run_main(SPEAKER_MODULE, "notjson{{{{")
        assert "error" in result

    def test_unknown_command_returns_error_key(self):
        result = _run_main(SPEAKER_MODULE, '{"command": "bogus_cmd"}')
        assert "error" in result
