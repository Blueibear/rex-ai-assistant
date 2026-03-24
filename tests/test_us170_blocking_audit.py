"""Tests for US-170: Audit and reduce blocking operations in the voice pipeline."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
VOICE_LOOP_SRC = REPO_ROOT / "rex" / "voice_loop.py"


def _src() -> str:
    return VOICE_LOOP_SRC.read_text(encoding="utf-8")


# ── No time.sleep in hot path ─────────────────────────────────────────────────


class TestNoTimeSleep:
    def test_no_time_sleep_in_voice_loop(self):
        src = _src()
        assert "time.sleep(" not in src

    def test_no_raw_sleep_call_in_run_method(self):
        src = _src()
        idx = src.index("async def run(self, max_interactions")
        # Grab ~5000 chars of run method body
        body = src[idx : idx + 5000]
        assert "time.sleep(" not in body


# ── Blocking I/O offloaded to threads ─────────────────────────────────────────


class TestBlockingIO:
    def _get_capture_fn_body(self) -> str:
        """Get the _capture() function body inside AsyncMicrophone._record."""
        src = _src()
        # _capture is a nested def inside async def _record
        idx = src.index("def _capture()")
        src.index("{", idx)
        # Use : start of function body
        src.index(":", idx)
        return src[idx : idx + 300]

    def test_sd_wait_is_in_thread_fn(self):
        """sd.wait() must be inside a function passed to asyncio.to_thread."""
        src = _src()
        # Find sd.wait() occurrences
        occurrences = [m.start() for m in re.finditer(r"sd\.wait\(\)", src)]
        # Each occurrence must be inside a def _capture or def _play function
        # that is itself passed to asyncio.to_thread
        for pos in occurrences:
            # Look back for nearest def
            before = src[:pos]
            # nearest function def should be _capture or _play
            last_def = before.rfind("def _")
            snippet = src[last_def : last_def + 50]
            assert (
                "def _capture" in snippet or "def _play" in snippet
            ), f"sd.wait() at position {pos} is not inside a thread function. Context: {snippet!r}"

    def test_play_obj_wait_done_in_thread_fn(self):
        """play_obj.wait_done() must be inside a thread function."""
        src = _src()
        occurrences = [m.start() for m in re.finditer(r"play_obj\.wait_done\(\)", src)]
        assert len(occurrences) > 0, "Expected play_obj.wait_done() calls"
        for pos in occurrences:
            before = src[:pos]
            last_def = before.rfind("def _")
            snippet = src[last_def : last_def + 50]
            assert "def _" in snippet

    def test_speak_edge_sf_read_not_on_event_loop(self):
        """sf.read in _speak_edge must be inside asyncio.to_thread, not bare."""
        src = _src()
        idx = src.index("async def _speak_edge")
        end_idx = src.index("async def _speak_windows", idx)
        edge_body = src[idx:end_idx]
        # If sf.read is present, it must be inside a def (thread function)
        if "sf.read(" in edge_body:
            # Check that sf.read is inside a nested def
            sf_read_pos = edge_body.index("sf.read(")
            before_sf = edge_body[:sf_read_pos]
            assert (
                "def _" in before_sf
            ), "sf.read() in _speak_edge is not inside a thread function (blocking event loop)"

    def test_speak_edge_sf_write_not_on_event_loop(self):
        """sf.write in _speak_edge must be inside asyncio.to_thread, not bare."""
        src = _src()
        idx = src.index("async def _speak_edge")
        end_idx = src.index("async def _speak_windows", idx)
        edge_body = src[idx:end_idx]
        if "sf.write(" in edge_body:
            sf_write_pos = edge_body.index("sf.write(")
            before_sf = edge_body[:sf_write_pos]
            assert (
                "def _" in before_sf
            ), "sf.write() in _speak_edge is not inside a thread function (blocking event loop)"

    def test_all_blocking_audio_in_to_thread(self):
        """asyncio.to_thread wraps all blocking audio calls."""
        src = _src()
        count = src.count("asyncio.to_thread(")
        assert count >= 4, f"Expected at least 4 asyncio.to_thread calls, found {count}"


# ── No synchronous transcription on event loop ────────────────────────────────


class TestSTTOffloaded:
    def test_transcribe_uses_to_thread(self):
        src = _src()
        idx = src.index("async def transcribe(")
        end_idx = src.index("class TextToSpeech", idx)
        body = src[idx:end_idx]
        assert "asyncio.to_thread" in body

    def test_no_whisper_call_outside_thread(self):
        """whisper model inference must happen inside asyncio.to_thread."""
        src = _src()
        idx = src.index("async def transcribe(")
        end_idx = src.index("class TextToSpeech", idx)
        transcribe_body = src[idx:end_idx]
        if "self._model.transcribe" in transcribe_body:
            call_pos = transcribe_body.index("self._model.transcribe")
            before = transcribe_body[:call_pos]
            assert "def _transcribe" in before


# ── XTTS synthesis offloaded ───────────────────────────────────────────────────


class TestXTTSOffloaded:
    def test_tts_to_file_inside_thread(self):
        src = _src()
        idx = src.index("async def _synthesize_and_play_chunk")
        end_idx = src.index("async def speak_streaming", idx)
        body = src[idx:end_idx]
        assert "asyncio.to_thread" in body
        tts_call_pos = body.index("tts_to_file")
        before = body[:tts_call_pos]
        assert "def _synthesize" in before


# ── AGENTS.md or commit documents the audit ───────────────────────────────────


class TestDocumentation:
    def test_agents_md_exists_or_voice_loop_has_audit_comment(self):
        """Either AGENTS.md documents the blocking audit or voice_loop.py has a comment."""
        agents_md = REPO_ROOT / "AGENTS.md"
        src = _src()
        has_comment = "asyncio.to_thread" in src and "blocking" in src.lower()
        agents_documented = (
            agents_md.exists() and "blocking" in agents_md.read_text(encoding="utf-8").lower()
        )
        assert (
            has_comment or agents_documented
        ), "Blocking audit not documented: add AGENTS.md entry or inline comment"
