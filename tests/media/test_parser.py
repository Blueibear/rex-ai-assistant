from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from rex.media.parser import MediaCommand, MediaCommandAction, parse_media_command


def test_play_extracts_query_only_without_target() -> None:
    command = parse_media_command("play some jazz")

    assert command == MediaCommand(action=MediaCommandAction.PLAY, query="some jazz")


def test_play_extracts_query_and_target_text() -> None:
    command = parse_media_command("play jazz in the kitchen")

    assert command is not None
    assert command.action is MediaCommandAction.PLAY
    assert command.query == "jazz"
    assert command.target_text == "kitchen"


def test_play_preserves_query_capitalization() -> None:
    command = parse_media_command("play The Beatles on the living room speaker")

    assert command is not None
    assert command.query == "The Beatles"
    assert command.target_text == "living room speaker"


@pytest.mark.parametrize(
    ("text", "action"),
    [
        ("pause", MediaCommandAction.PAUSE),
        ("pause the music", MediaCommandAction.PAUSE),
        ("resume", MediaCommandAction.RESUME),
        ("unpause", MediaCommandAction.RESUME),
        ("continue the playback", MediaCommandAction.RESUME),
        ("stop", MediaCommandAction.STOP),
        ("stop the music", MediaCommandAction.STOP),
        ("next", MediaCommandAction.NEXT),
        ("skip", MediaCommandAction.NEXT),
        ("skip track", MediaCommandAction.NEXT),
        ("previous", MediaCommandAction.PREVIOUS),
        ("go back", MediaCommandAction.PREVIOUS),
        ("back", MediaCommandAction.PREVIOUS),
    ],
)
def test_simple_transport_actions_parse_without_target(
    text: str, action: MediaCommandAction
) -> None:
    command = parse_media_command(text)

    assert command == MediaCommand(action=action)


def test_pause_extracts_target_text() -> None:
    command = parse_media_command("pause the kitchen speaker")

    assert command == MediaCommand(action=MediaCommandAction.PAUSE, target_text="kitchen speaker")


def test_next_extracts_target_text() -> None:
    command = parse_media_command("skip track in the den")

    assert command == MediaCommand(action=MediaCommandAction.NEXT, target_text="den")


def test_set_volume_extracts_bounded_level() -> None:
    command = parse_media_command("set volume to 42")

    assert command == MediaCommand(action=MediaCommandAction.SET_VOLUME, level=42)


def test_set_volume_extracts_level_and_target() -> None:
    command = parse_media_command("volume to 15 in the kitchen")

    assert command == MediaCommand(
        action=MediaCommandAction.SET_VOLUME, level=15, target_text="kitchen"
    )


@pytest.mark.parametrize("text", ["set volume to 101", "volume to -1", "set volume to 1000"])
def test_set_volume_rejects_out_of_bounds_level(text: str) -> None:
    assert parse_media_command(text) is None


def test_mute_parses_without_level() -> None:
    command = parse_media_command("mute the kitchen speaker")

    assert command == MediaCommand(action=MediaCommandAction.MUTE, target_text="kitchen speaker")


def test_unmute_without_level_leaves_level_none() -> None:
    command = parse_media_command("unmute the speaker")

    assert command == MediaCommand(action=MediaCommandAction.UNMUTE, target_text=None)


def test_unmute_with_explicit_level() -> None:
    command = parse_media_command("unmute to 30 in the den")

    assert command == MediaCommand(action=MediaCommandAction.UNMUTE, level=30, target_text="den")


@pytest.mark.parametrize(
    "text",
    [
        "what's playing",
        "what is playing",
        "is it playing",
        "is the music playing",
        "which song is playing",
        "what track is this",
    ],
)
def test_query_state_phrases_parse(text: str) -> None:
    command = parse_media_command(text)

    assert command is not None
    assert command.action is MediaCommandAction.QUERY_STATE


def test_query_state_extracts_target_text() -> None:
    command = parse_media_command("what's playing in the kitchen")

    assert command == MediaCommand(action=MediaCommandAction.QUERY_STATE, target_text="kitchen")


@pytest.mark.parametrize(
    "text",
    [
        "move it to the living room",
        "send it to the living room",
        "transfer it to the living room",
        "play it in the living room",
        "move this to the living room",
    ],
)
def test_transfer_phrases_extract_target_only(text: str) -> None:
    command = parse_media_command(text)

    assert command == MediaCommand(action=MediaCommandAction.TRANSFER, target_text="living room")


def test_transfer_never_resolves_authority_only_raw_text() -> None:
    command = parse_media_command("move it to sonos:RINCON_2")

    assert command is not None
    assert command.action is MediaCommandAction.TRANSFER
    assert command.target_text == "sonos:RINCON_2"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "hello there",
        "what time is it",
        "play",
        "play it",
        "set a timer for five minutes",
    ],
)
def test_unsupported_or_empty_text_returns_none(text: str) -> None:
    assert parse_media_command(text) is None


def test_parser_normalizes_whitespace_case_and_punctuation() -> None:
    command = parse_media_command("  PAUSE   the   MUSIC!  ")

    assert command == MediaCommand(action=MediaCommandAction.PAUSE)


def test_media_command_is_immutable_and_field_order_matches_contract() -> None:
    command = MediaCommand(
        action=MediaCommandAction.SET_VOLUME, query=None, target_text="kitchen", level=50
    )

    assert command.action is MediaCommandAction.SET_VOLUME
    assert command.target_text == "kitchen"
    assert command.level == 50
    with pytest.raises(FrozenInstanceError):
        command.level = 10  # type: ignore[misc]
