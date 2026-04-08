"""Tests for US-023: room context priority resolution."""

from rex.context.room import RoomContext


def test_explicit_wins_over_all():
    ctx = RoomContext(config_default="lounge")
    ctx.set_speaker_origin("kitchen")
    ctx.set_last_active("hallway")
    ctx.set_explicit("bedroom")
    assert ctx.current_room == "bedroom"


def test_speaker_origin_wins_over_last_active_and_default():
    ctx = RoomContext(config_default="lounge")
    ctx.set_last_active("hallway")
    ctx.set_speaker_origin("kitchen")
    assert ctx.current_room == "kitchen"


def test_last_active_wins_over_config_default():
    ctx = RoomContext(config_default="lounge")
    ctx.set_last_active("hallway")
    assert ctx.current_room == "hallway"


def test_config_default_used_when_all_others_none():
    ctx = RoomContext(config_default="lounge")
    assert ctx.current_room == "lounge"


def test_none_when_nothing_set():
    ctx = RoomContext()
    assert ctx.current_room is None


def test_clear_explicit_falls_back_to_speaker_origin():
    ctx = RoomContext(config_default="lounge")
    ctx.set_speaker_origin("kitchen")
    ctx.set_explicit("bedroom")
    assert ctx.current_room == "bedroom"
    ctx.clear_explicit()
    assert ctx.current_room == "kitchen"


def test_set_config_default_updates_fallback():
    ctx = RoomContext()
    assert ctx.current_room is None
    ctx.set_config_default("study")
    assert ctx.current_room == "study"


def test_priority_order_all_sources_set():
    ctx = RoomContext(config_default="d")
    ctx.set_last_active("c")
    ctx.set_speaker_origin("b")
    ctx.set_explicit("a")
    assert ctx.current_room == "a"
    ctx.clear_explicit()
    assert ctx.current_room == "b"
    ctx.set_speaker_origin(None)
    assert ctx.current_room == "c"
    ctx.set_last_active(None)
    assert ctx.current_room == "d"
