"""Tests for US-022: Wire Music Assistant commands to assistant tool routing."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rex.media.parser import MediaCommandAction
from rex.music_handler import (
    _PAUSE_PATTERNS,
    _RESUME_PATTERNS,
    _SKIP_PATTERNS,
    MusicHandler,
    _match_play,
    _match_room_command,
    _match_volume,
)
from rex.tools.registry import get_default_registry

# ---------------------------------------------------------------------------
# Pattern matching unit tests
# ---------------------------------------------------------------------------


class TestMatchPlay:
    def test_simple_play(self):
        matched, query, room = _match_play("play Shape of You")
        assert matched
        assert query == "Shape of You"
        assert room is None

    def test_play_with_room(self):
        matched, query, room = _match_play("play jazz in the kitchen")
        assert matched
        assert "jazz" in query
        assert room == "kitchen"

    def test_play_some_music(self):
        matched, query, room = _match_play("play some music")
        assert matched
        assert query == "some music"

    def test_no_match(self):
        matched, query, room = _match_play("pause the music")
        assert not matched


class TestMatchPause:
    def test_pause_bare(self):
        matched, room = _match_room_command(_PAUSE_PATTERNS, "pause")
        assert matched
        assert room is None

    def test_pause_music(self):
        matched, room = _match_room_command(_PAUSE_PATTERNS, "pause the music")
        assert matched
        assert room is None

    def test_pause_with_room(self):
        matched, room = _match_room_command(_PAUSE_PATTERNS, "pause in the living room")
        assert matched
        assert room == "living room"

    def test_stop_music(self):
        matched, room = _match_room_command(_PAUSE_PATTERNS, "stop the music")
        assert matched

    def test_no_match(self):
        matched, room = _match_room_command(_PAUSE_PATTERNS, "play some music")
        assert not matched


class TestMatchResume:
    def test_resume_bare(self):
        matched, room = _match_room_command(_RESUME_PATTERNS, "resume")
        assert matched
        assert room is None

    def test_continue_playing(self):
        matched, room = _match_room_command(_RESUME_PATTERNS, "continue playing")
        assert matched

    def test_no_match(self):
        matched, room = _match_room_command(_RESUME_PATTERNS, "skip this song")
        assert not matched


class TestMatchSkip:
    def test_skip_bare(self):
        matched, room = _match_room_command(_SKIP_PATTERNS, "skip")
        assert matched
        assert room is None

    def test_next_song(self):
        matched, room = _match_room_command(_SKIP_PATTERNS, "next song")
        assert matched

    def test_skip_this_song(self):
        matched, room = _match_room_command(_SKIP_PATTERNS, "skip this song")
        assert matched

    def test_no_match(self):
        matched, room = _match_room_command(_SKIP_PATTERNS, "volume 50")
        assert not matched


class TestMatchVolume:
    def test_volume_level(self):
        matched, level, room = _match_volume("volume 70")
        assert matched
        assert level == 70
        assert room is None

    def test_set_volume_to(self):
        matched, level, room = _match_volume("set volume to 50")
        assert matched
        assert level == 50

    def test_volume_with_room(self):
        matched, level, room = _match_volume("volume 80 in the kitchen")
        assert matched
        assert level == 80
        assert room == "kitchen"

    def test_no_match(self):
        matched, level, room = _match_volume("play jazz")
        assert not matched


# ---------------------------------------------------------------------------
# MusicHandler.handle() integration tests
# ---------------------------------------------------------------------------


class TestMusicHandler:
    def _make_handler(self, **mock_methods):
        client = MagicMock()
        for name, rv in mock_methods.items():
            getattr(client, name).return_value = rv
        return MusicHandler(client), client

    def test_compat_handler_parses_without_mutating_provider(self):
        handler, client = self._make_handler(play={})

        command = handler.handle("play jazz in the kitchen")

        assert command is not None
        assert command.action is MediaCommandAction.PLAY
        assert command.query == "jazz"
        assert command.target_text == "kitchen"
        client.play.assert_not_called()

    @pytest.mark.parametrize(
        ("transcript", "action", "target", "level"),
        [
            ("please play jazz in the kitchen", MediaCommandAction.PLAY, "kitchen", None),
            ("continue playing", MediaCommandAction.RESUME, None, None),
            ("next song", MediaCommandAction.NEXT, None, None),
            ("skip this song", MediaCommandAction.NEXT, None, None),
            ("turn volume up to 80", MediaCommandAction.SET_VOLUME, None, 80),
        ],
    )
    def test_legacy_phrases_parse_through_canonical_media_command(
        self, transcript, action, target, level
    ):
        handler, client = self._make_handler()

        command = handler.handle(transcript)

        assert command is not None
        assert command.action is action
        assert command.target_text == target
        assert command.level == level
        assert client.method_calls == []

    @pytest.mark.parametrize(
        ("transcript", "action"),
        [
            ("play Shape of You", MediaCommandAction.PLAY),
            ("play jazz in the kitchen", MediaCommandAction.PLAY),
            ("pause", MediaCommandAction.PAUSE),
            ("resume", MediaCommandAction.RESUME),
            ("skip", MediaCommandAction.NEXT),
            ("set volume to 60", MediaCommandAction.SET_VOLUME),
        ],
    )
    def test_handle_is_parse_only_and_never_calls_provider(self, transcript, action):
        handler, client = self._make_handler()

        command = handler.handle(transcript)

        assert command is not None
        assert command.action is action
        assert client.method_calls == []

    def test_non_music_transcript_returns_none(self):
        handler, client = self._make_handler()
        assert handler.handle("what is the weather today") is None
        assert client.method_calls == []


# ---------------------------------------------------------------------------
# Tool catalog tests
# ---------------------------------------------------------------------------


class TestMusicToolRegistry:
    def test_legacy_music_tools_are_retired_after_canonical_cutover(self):
        registry = get_default_registry()
        names = {t.name for t in registry.all_tools()}
        legacy = {
            "music_play",
            "music_pause",
            "music_resume",
            "music_skip",
            "music_volume",
        }
        assert names.isdisjoint(legacy)
        assert {"media_read", "media_manage"}.issubset(names)


# ---------------------------------------------------------------------------
# Assistant.generate_reply integration test
# ---------------------------------------------------------------------------


class TestAssistantMusicRouting:
    def test_assistant_startup_has_no_direct_music_handler(self, monkeypatch, tmp_path):
        import rex.assistant as assistant_module
        from rex.assistant import Assistant

        class DummyLanguageModel:
            def __init__(self, *args, **kwargs):
                pass

            def generate(self, *args, **kwargs):
                return "ok"

        monkeypatch.setattr(assistant_module, "LanguageModel", DummyLanguageModel)
        assistant = Assistant(transcripts_dir=tmp_path, user_id="default")

        assert assistant._music_handler is None
        assert assistant._media_service is not None
