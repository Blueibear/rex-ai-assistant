"""Tests for US-024: speaker origin detection in RoomContext."""

from __future__ import annotations

from rex.context.room import RoomContext

# ---------------------------------------------------------------------------
# MQTT topic parsing
# ---------------------------------------------------------------------------


def test_mqtt_topic_sets_speaker_origin():
    ctx = RoomContext()
    result = ctx.set_speaker_origin_from_topic("rex/audio/kitchen")
    assert result is True
    assert ctx._speaker_origin == "kitchen"


def test_mqtt_topic_current_room_resolves():
    ctx = RoomContext(config_default="living_room")
    ctx.set_speaker_origin_from_topic("rex/audio/bedroom")
    assert ctx.current_room == "bedroom"


def test_mqtt_topic_non_matching_returns_false():
    ctx = RoomContext()
    result = ctx.set_speaker_origin_from_topic("rex/voice/kitchen")
    assert result is False
    assert ctx._speaker_origin is None


def test_mqtt_topic_trailing_slash_does_not_match():
    ctx = RoomContext()
    result = ctx.set_speaker_origin_from_topic("rex/audio/kitchen/extra")
    assert result is False


def test_mqtt_topic_empty_room_segment_does_not_match():
    ctx = RoomContext()
    result = ctx.set_speaker_origin_from_topic("rex/audio/")
    assert result is False


# ---------------------------------------------------------------------------
# Device-to-room mapping
# ---------------------------------------------------------------------------


def test_device_mapping_sets_speaker_origin():
    ctx = RoomContext()
    mapping = {"mic_kitchen": "kitchen", "mic_office": "office"}
    result = ctx.set_speaker_origin_from_device("mic_kitchen", mapping)
    assert result is True
    assert ctx._speaker_origin == "kitchen"


def test_device_mapping_unknown_device_returns_false():
    ctx = RoomContext()
    mapping = {"mic_kitchen": "kitchen"}
    result = ctx.set_speaker_origin_from_device("mic_living_room", mapping)
    assert result is False
    assert ctx._speaker_origin is None


def test_device_mapping_empty_map_returns_false():
    ctx = RoomContext()
    result = ctx.set_speaker_origin_from_device("mic_kitchen", {})
    assert result is False


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------


def test_no_mapping_falls_back_to_config_default():
    ctx = RoomContext(config_default="lounge")
    # Neither MQTT nor device sets anything
    assert ctx.current_room == "lounge"


def test_explicit_overrides_mqtt_origin():
    ctx = RoomContext()
    ctx.set_speaker_origin_from_topic("rex/audio/kitchen")
    ctx.set_explicit("office")
    assert ctx.current_room == "office"


def test_mqtt_origin_overrides_last_active():
    ctx = RoomContext()
    ctx.set_last_active("bedroom")
    ctx.set_speaker_origin_from_topic("rex/audio/kitchen")
    assert ctx.current_room == "kitchen"
