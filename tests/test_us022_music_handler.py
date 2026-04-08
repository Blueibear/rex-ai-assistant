"""Tests for US-022: Wire Music Assistant commands to assistant tool routing."""

from __future__ import annotations

from unittest.mock import MagicMock

from rex.assistant_errors import IntegrationNotConfiguredError
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

    def test_handle_play_routes_to_client(self):
        handler, client = self._make_handler(play={})
        response = handler.handle("play Shape of You")
        client.play.assert_called_once_with("Shape of You", room=None)
        assert "Shape of You" in response

    def test_handle_play_with_room(self):
        handler, client = self._make_handler(play={})
        response = handler.handle("play jazz in the kitchen")
        client.play.assert_called_once()
        _, kwargs = client.play.call_args
        assert kwargs["room"] == "kitchen"
        assert "kitchen" in response

    def test_handle_pause_routes_to_client(self):
        handler, client = self._make_handler(pause={})
        response = handler.handle("pause")
        client.pause.assert_called_once()
        assert response == "Music paused."

    def test_handle_resume_routes_to_client(self):
        handler, client = self._make_handler(resume={})
        response = handler.handle("resume")
        client.resume.assert_called_once()
        assert response == "Music resumed."

    def test_handle_skip_routes_to_client(self):
        handler, client = self._make_handler(skip={})
        response = handler.handle("skip")
        client.skip.assert_called_once()
        assert "next track" in response

    def test_handle_volume_routes_to_client(self):
        handler, client = self._make_handler(set_volume={})
        response = handler.handle("set volume to 60")
        client.set_volume.assert_called_once_with(60, room=None)
        assert "60" in response

    def test_not_configured_returns_friendly_message(self):
        client = MagicMock()
        client.play.side_effect = IntegrationNotConfiguredError("not configured")
        handler = MusicHandler(client)
        response = handler.handle("play jazz")
        assert response == "Music Assistant is not set up."

    def test_non_music_transcript_returns_none(self):
        handler, _ = self._make_handler()
        assert handler.handle("what is the weather today") is None

    def test_pause_not_configured_returns_friendly_message(self):
        client = MagicMock()
        client.pause.side_effect = IntegrationNotConfiguredError("not configured")
        handler = MusicHandler(client)
        assert handler.handle("pause") == "Music Assistant is not set up."

    def test_skip_not_configured_returns_friendly_message(self):
        client = MagicMock()
        client.skip.side_effect = IntegrationNotConfiguredError("not configured")
        handler = MusicHandler(client)
        assert handler.handle("skip") == "Music Assistant is not set up."


# ---------------------------------------------------------------------------
# Tool catalog tests
# ---------------------------------------------------------------------------


class TestMusicToolRegistry:
    def test_music_tools_in_catalog(self):
        registry = get_default_registry()
        names = {t.name for t in registry.all_tools()}
        assert "music_play" in names
        assert "music_pause" in names
        assert "music_resume" in names
        assert "music_skip" in names
        assert "music_volume" in names

    def test_music_tools_require_music_assistant_url(self):
        registry = get_default_registry()
        music_tools = [t for t in registry.all_tools() if t.name.startswith("music_")]
        for tool in music_tools:
            assert "music_assistant_url" in tool.requires_config

    def test_music_tools_unavailable_when_not_configured(self):
        registry = get_default_registry()

        class FakeCfg:
            music_assistant_url = None

        available_names = {t.name for t in registry.available_tools(FakeCfg())}
        assert "music_play" not in available_names

    def test_music_tools_available_when_configured(self):
        registry = get_default_registry()

        class FakeCfg:
            music_assistant_url = "http://localhost:8095"

        available_names = {t.name for t in registry.available_tools(FakeCfg())}
        assert "music_play" in available_names
        assert "music_pause" in available_names


# ---------------------------------------------------------------------------
# Assistant.generate_reply integration test
# ---------------------------------------------------------------------------


class TestAssistantMusicRouting:
    def test_generate_reply_routes_music_intent(self):
        """generate_reply() must route play commands to MusicHandler, not LLM."""
        import asyncio
        import pathlib

        from rex.assistant import Assistant
        from rex.music_handler import MusicHandler

        mock_client = MagicMock()
        mock_client.play.return_value = {}

        # Build a minimal Assistant instance without full __init__ to avoid
        # external service connections; inject only what generate_reply() needs.
        assistant = Assistant.__new__(Assistant)
        assistant._music_handler = MusicHandler(mock_client)
        assistant._shopping_list_handler = None
        assistant._skill_trainer = None
        assistant._skill_registry = None
        assistant._skill_router = None
        assistant._tool_dispatcher = None
        assistant._response_cache = None
        assistant._ha_bridge = None
        assistant._history = []
        assistant._history_limit = 10
        assistant._history_store = None
        assistant._transcripts_dir = pathlib.Path("/tmp")
        assistant._user_id = "test"
        assistant._record_completion = lambda t, c: None  # type: ignore[method-assign]
        assistant._router = MagicMock()
        assistant._router.classify.return_value = "general"
        assistant._router.resolve_model.return_value = ""
        assistant._llm = MagicMock()
        assistant._settings = MagicMock()
        assistant._followup_engine = None
        assistant._pending_followup = None

        result = asyncio.run(assistant.generate_reply("play Shape of You"))

        assert "Shape of You" in result
        mock_client.play.assert_called_once()
