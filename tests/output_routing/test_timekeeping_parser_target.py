from __future__ import annotations

from datetime import UTC, datetime

from rex.timekeeping.parser import parse_timekeeping_command

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
TZ = "America/Chicago"


def test_alarm_parser_extracts_output_target_clause() -> None:
    command = parse_timekeeping_command(
        "set an alarm for 7 am and play it on the bedroom speaker",
        user_timezone=TZ,
        now=NOW,
    )

    assert command is not None
    assert command.action == "create_alarm"
    assert command.target_text == "bedroom speaker"


def test_timer_parser_extracts_room_target_without_corrupting_name() -> None:
    command = parse_timekeeping_command(
        "set a 10 minute pasta timer in the kitchen",
        user_timezone=TZ,
        now=NOW,
    )

    assert command is not None
    assert command.action == "create_timer"
    assert command.reference == "pasta"
    assert command.target_text == "kitchen"
