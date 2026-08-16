from __future__ import annotations

from datetime import UTC, date, datetime, time

import pytest

from rex.timekeeping.parser import parse_timekeeping_command

NOW = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)
TZ = "America/Chicago"


@pytest.mark.parametrize(
    ("text", "seconds", "name"),
    [
        ("set a 10-minute timer", 600, None),
        ("set a 20-minute laundry timer", 1200, "laundry"),
        ("start a pasta timer for 5 minutes", 300, "pasta"),
        ("set a timer for 45 seconds", 45, None),
        ("create a 2 hour nap timer", 7200, "nap"),
    ],
)
def test_parse_timer_creation(text, seconds, name) -> None:
    command = parse_timekeeping_command(text, user_timezone=TZ, now=NOW)
    assert command is not None
    assert command.action == "create_timer"
    assert command.duration_seconds == seconds
    assert command.reference == name


def test_parse_timer_management_commands() -> None:
    assert (
        parse_timekeeping_command("pause the pasta timer", user_timezone=TZ, now=NOW).action
        == "pause_timer"
    )
    assert (
        parse_timekeeping_command("resume pasta timer", user_timezone=TZ, now=NOW).action
        == "resume_timer"
    )
    assert (
        parse_timekeeping_command("cancel the laundry timer", user_timezone=TZ, now=NOW).reference
        == "laundry"
    )

    adjusted = parse_timekeeping_command(
        "add five minutes to the pasta timer", user_timezone=TZ, now=NOW
    )
    assert adjusted.action == "adjust_timer"
    assert adjusted.reference == "pasta"
    assert adjusted.delta_seconds == 300

    renamed = parse_timekeeping_command(
        "rename the pasta timer to sauce", user_timezone=TZ, now=NOW
    )
    assert renamed.action == "rename_timer"
    assert renamed.reference == "pasta"
    assert renamed.new_name == "sauce"


def test_parse_timer_queries() -> None:
    all_timers = parse_timekeeping_command(
        "how much time is left on my timers", user_timezone=TZ, now=NOW
    )
    assert all_timers.action == "list_timers"

    pasta = parse_timekeeping_command(
        "how much time is left on the pasta timer", user_timezone=TZ, now=NOW
    )
    assert pasta.action == "query_timer"
    assert pasta.reference == "pasta"


def test_parse_one_shot_alarm_for_tomorrow_morning() -> None:
    command = parse_timekeeping_command(
        "set an alarm for 7:00 tomorrow morning", user_timezone=TZ, now=NOW
    )
    assert command.action == "create_alarm"
    assert command.alarm_time == time(7, 0)
    assert command.alarm_date == date(2026, 8, 17)
    assert command.weekdays == ()


def test_parse_weekday_and_selected_day_alarm_recurrence() -> None:
    weekday = parse_timekeeping_command("wake me at 7:00 every weekday", user_timezone=TZ, now=NOW)
    assert weekday.action == "create_alarm"
    assert weekday.alarm_time == time(7, 0)
    assert weekday.weekdays == (0, 1, 2, 3, 4)

    selected = parse_timekeeping_command(
        "set an alarm for 6:30 am every monday wednesday friday",
        user_timezone=TZ,
        now=NOW,
    )
    assert selected.alarm_time == time(6, 30)
    assert selected.weekdays == (0, 2, 4)


def test_parse_alarm_management() -> None:
    snooze = parse_timekeeping_command(
        "snooze that alarm for 10 minutes", user_timezone=TZ, now=NOW
    )
    assert snooze.action == "snooze_alarm"
    assert snooze.duration_seconds == 600
    assert snooze.reference is None

    assert (
        parse_timekeeping_command("dismiss the morning alarm", user_timezone=TZ, now=NOW).action
        == "dismiss_alarm"
    )
    assert (
        parse_timekeeping_command("disable my morning alarm", user_timezone=TZ, now=NOW).action
        == "disable_alarm"
    )
    assert (
        parse_timekeeping_command("enable morning alarm", user_timezone=TZ, now=NOW).action
        == "enable_alarm"
    )
    assert (
        parse_timekeeping_command("cancel the morning alarm", user_timezone=TZ, now=NOW).action
        == "cancel_alarm"
    )


def test_parser_rejects_unrelated_or_invalid_timekeeping_text() -> None:
    assert parse_timekeeping_command("tell me a joke", user_timezone=TZ, now=NOW) is None
    assert parse_timekeeping_command("set a zero minute timer", user_timezone=TZ, now=NOW) is None


def test_parse_alarm_edit_and_rename() -> None:
    moved = parse_timekeeping_command(
        "change the morning alarm to 8:15 am", user_timezone=TZ, now=NOW
    )
    assert moved.action == "edit_alarm"
    assert moved.reference == "morning"
    assert moved.alarm_time == time(8, 15)

    renamed = parse_timekeeping_command(
        "rename morning alarm to workday", user_timezone=TZ, now=NOW
    )
    assert renamed.action == "edit_alarm"
    assert renamed.reference == "morning"
    assert renamed.new_name == "workday"
