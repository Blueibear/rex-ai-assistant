"""Tests for US-055: Safety layer for computer control.

Covers:
- classify_action: safe, dangerous, unknown
- SafetyLayer.requires_confirmation: always, dangerous_only, never modes
- SafetyLayer.check: safe bypass, dangerous with confirm, dangerous denied (no fn),
  always-confirm for safe action, never-confirm bypass
- AppConfig.computer_control_confirmation field exists with default "dangerous_only"
- Typecheck
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# classify_action
# ---------------------------------------------------------------------------


class TestClassifyAction:
    def test_read_file_is_safe(self) -> None:
        from rex.computers.safety import ActionType, classify_action

        assert classify_action("read_file") == ActionType.safe

    def test_list_dir_is_safe(self) -> None:
        from rex.computers.safety import ActionType, classify_action

        assert classify_action("list_dir") == ActionType.safe

    def test_search_files_is_safe(self) -> None:
        from rex.computers.safety import ActionType, classify_action

        assert classify_action("search_files") == ActionType.safe

    def test_summarize_file_is_safe(self) -> None:
        from rex.computers.safety import ActionType, classify_action

        assert classify_action("summarize_file") == ActionType.safe

    def test_write_file_is_dangerous(self) -> None:
        from rex.computers.safety import ActionType, classify_action

        assert classify_action("write_file") == ActionType.dangerous

    def test_delete_file_is_dangerous(self) -> None:
        from rex.computers.safety import ActionType, classify_action

        assert classify_action("delete_file") == ActionType.dangerous

    def test_execute_command_is_dangerous(self) -> None:
        from rex.computers.safety import ActionType, classify_action

        assert classify_action("execute_command") == ActionType.dangerous

    def test_launch_app_is_dangerous(self) -> None:
        from rex.computers.safety import ActionType, classify_action

        assert classify_action("launch_app") == ActionType.dangerous

    def test_unknown_action_is_unknown(self) -> None:
        from rex.computers.safety import ActionType, classify_action

        assert classify_action("do_something_weird") == ActionType.unknown

    def test_case_insensitive(self) -> None:
        from rex.computers.safety import ActionType, classify_action

        assert classify_action("READ_FILE") == ActionType.safe
        assert classify_action("Write_File") == ActionType.dangerous


# ---------------------------------------------------------------------------
# SafetyLayer.requires_confirmation
# ---------------------------------------------------------------------------


class TestRequiresConfirmation:
    def test_never_mode_safe_action(self) -> None:
        from rex.computers.safety import SafetyLayer

        layer = SafetyLayer(mode="never")
        assert layer.requires_confirmation("read_file") is False

    def test_never_mode_dangerous_action(self) -> None:
        from rex.computers.safety import SafetyLayer

        layer = SafetyLayer(mode="never")
        assert layer.requires_confirmation("write_file") is False

    def test_always_mode_safe_action(self) -> None:
        from rex.computers.safety import SafetyLayer

        layer = SafetyLayer(mode="always")
        assert layer.requires_confirmation("read_file") is True

    def test_always_mode_dangerous_action(self) -> None:
        from rex.computers.safety import SafetyLayer

        layer = SafetyLayer(mode="always")
        assert layer.requires_confirmation("write_file") is True

    def test_dangerous_only_safe_action(self) -> None:
        from rex.computers.safety import SafetyLayer

        layer = SafetyLayer(mode="dangerous_only")
        assert layer.requires_confirmation("read_file") is False

    def test_dangerous_only_dangerous_action(self) -> None:
        from rex.computers.safety import SafetyLayer

        layer = SafetyLayer(mode="dangerous_only")
        assert layer.requires_confirmation("write_file") is True

    def test_dangerous_only_unknown_action(self) -> None:
        from rex.computers.safety import SafetyLayer

        layer = SafetyLayer(mode="dangerous_only")
        assert layer.requires_confirmation("unknown_op") is True

    def test_invalid_mode_raises(self) -> None:
        from rex.computers.safety import SafetyLayer

        with pytest.raises(ValueError, match="Invalid confirmation mode"):
            SafetyLayer(mode="maybe")


# ---------------------------------------------------------------------------
# SafetyLayer.check — confirmation flow
# ---------------------------------------------------------------------------


class TestCheck:
    def test_safe_action_proceeds_without_confirm_fn(self) -> None:
        from rex.computers.safety import SafetyLayer

        layer = SafetyLayer(mode="dangerous_only")
        assert layer.check("read_file") is True

    def test_dangerous_action_denied_when_no_confirm_fn(self) -> None:
        from rex.computers.safety import SafetyLayer

        layer = SafetyLayer(mode="dangerous_only")
        assert layer.check("write_file") is False

    def test_dangerous_action_allowed_when_confirm_fn_returns_true(self) -> None:
        from rex.computers.safety import SafetyLayer

        layer = SafetyLayer(mode="dangerous_only")
        assert layer.check("write_file", confirm_fn=lambda _: True) is True

    def test_dangerous_action_denied_when_confirm_fn_returns_false(self) -> None:
        from rex.computers.safety import SafetyLayer

        layer = SafetyLayer(mode="dangerous_only")
        assert layer.check("write_file", confirm_fn=lambda _: False) is False

    def test_confirm_fn_receives_description(self) -> None:
        from rex.computers.safety import SafetyLayer

        captured: list[str] = []

        def my_confirm(desc: str) -> bool:
            captured.append(desc)
            return True

        layer = SafetyLayer(mode="dangerous_only")
        layer.check("write_file", description="Write to /home/user/doc.txt", confirm_fn=my_confirm)
        assert captured == ["Write to /home/user/doc.txt"]

    def test_never_mode_no_confirm_needed_even_for_dangerous(self) -> None:
        from rex.computers.safety import SafetyLayer

        layer = SafetyLayer(mode="never")
        assert layer.check("delete_file") is True

    def test_always_mode_safe_action_still_needs_confirm(self) -> None:
        from rex.computers.safety import SafetyLayer

        layer = SafetyLayer(mode="always")
        # No confirm_fn → denied
        assert layer.check("read_file") is False

    def test_always_mode_safe_action_confirmed(self) -> None:
        from rex.computers.safety import SafetyLayer

        layer = SafetyLayer(mode="always")
        assert layer.check("read_file", confirm_fn=lambda _: True) is True


# ---------------------------------------------------------------------------
# AppConfig field
# ---------------------------------------------------------------------------


class TestAppConfigField:
    def test_default_is_dangerous_only(self) -> None:
        from rex.config import AppConfig

        cfg = AppConfig()
        assert cfg.computer_control_confirmation == "dangerous_only"

    def test_field_can_be_set_to_always(self) -> None:
        from rex.config import AppConfig

        cfg = AppConfig()
        cfg.computer_control_confirmation = "always"
        assert cfg.computer_control_confirmation == "always"

    def test_field_can_be_set_to_never(self) -> None:
        from rex.config import AppConfig

        cfg = AppConfig()
        cfg.computer_control_confirmation = "never"
        assert cfg.computer_control_confirmation == "never"
