"""Tests for US-167: Voice pipeline latency instrumentation and baseline doc."""

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
LATENCY_MODULE = REPO_ROOT / "rex" / "voice_latency.py"
VOICE_LOOP = REPO_ROOT / "rex" / "voice_loop.py"
BASELINE_DOC = REPO_ROOT / "docs" / "performance-baseline.md"


def _latency_src() -> str:
    return LATENCY_MODULE.read_text(encoding="utf-8")


def _voice_loop_src() -> str:
    return VOICE_LOOP.read_text(encoding="utf-8")


def _baseline_doc() -> str:
    return BASELINE_DOC.read_text(encoding="utf-8")


# ── Latency module ─────────────────────────────────────────────────────────────


class TestLatencyModule:
    def test_module_exists(self):
        assert LATENCY_MODULE.exists()

    def test_tracker_class_exists(self):
        assert "VoiceLatencyTracker" in _latency_src()

    def test_mark_method_exists(self):
        assert "def mark" in _latency_src()

    def test_elapsed_method_exists(self):
        assert "def elapsed" in _latency_src()

    def test_summary_method_exists(self):
        assert "def summary" in _latency_src()

    def test_log_summary_method_exists(self):
        assert "def log_summary" in _latency_src()

    def test_uses_perf_counter(self):
        assert "perf_counter" in _latency_src()

    def test_tracks_stt_stages(self):
        src = _latency_src()
        assert "stt_start" in src
        assert "stt_end" in src

    def test_tracks_llm_stages(self):
        src = _latency_src()
        assert "llm_start" in src
        assert "llm_end" in src

    def test_tracks_tts_stages(self):
        src = _latency_src()
        assert "tts_synthesis_start" in src

    def test_tracks_playback_start(self):
        assert "playback_start" in _latency_src()

    def test_tracker_functional(self):
        import time

        from rex.voice_latency import VoiceLatencyTracker

        tracker = VoiceLatencyTracker()
        tracker.mark("stt_start")
        time.sleep(0.01)
        tracker.mark("stt_end")
        elapsed = tracker.elapsed("stt_start", "stt_end")
        assert elapsed is not None
        assert elapsed >= 0.009

    def test_elapsed_returns_none_for_missing_stage(self):
        from rex.voice_latency import VoiceLatencyTracker

        tracker = VoiceLatencyTracker()
        tracker.mark("stt_start")
        assert tracker.elapsed("stt_start", "stt_end") is None

    def test_summary_returns_dict(self):
        import time

        from rex.voice_latency import VoiceLatencyTracker

        tracker = VoiceLatencyTracker()
        tracker.mark("stt_start")
        time.sleep(0.005)
        tracker.mark("stt_end")
        tracker.mark("llm_start")
        time.sleep(0.005)
        tracker.mark("llm_end")
        tracker.mark("tts_synthesis_start")
        time.sleep(0.005)
        tracker.mark("tts_synthesis_end")
        tracker.mark("playback_start")

        s = tracker.summary()
        assert isinstance(s, dict)
        assert "stt_s" in s
        assert "llm_s" in s
        assert "tts_synthesis_s" in s
        assert "total_s" in s
        assert s["stt_s"] >= 0.004
        assert s["total_s"] >= 0.014


# ── Voice loop instrumentation ────────────────────────────────────────────────


class TestVoiceLoopInstrumentation:
    def test_voice_latency_imported_in_run(self):
        src = _voice_loop_src()
        assert "voice_latency" in src or "VoiceLatencyTracker" in src

    def test_stt_start_mark_in_run(self):
        src = _voice_loop_src()
        assert 'tracker.mark("stt_start")' in src or "mark('stt_start')" in src

    def test_stt_end_mark_in_run(self):
        src = _voice_loop_src()
        assert 'tracker.mark("stt_end")' in src or "mark('stt_end')" in src

    def test_llm_start_mark_in_run(self):
        src = _voice_loop_src()
        assert 'tracker.mark("llm_start")' in src

    def test_llm_end_mark_in_run(self):
        src = _voice_loop_src()
        assert 'tracker.mark("llm_end")' in src

    def test_tts_synthesis_start_mark_in_run(self):
        src = _voice_loop_src()
        assert 'tracker.mark("tts_synthesis_start")' in src

    def test_playback_start_mark_in_run(self):
        src = _voice_loop_src()
        assert 'tracker.mark("playback_start")' in src

    def test_log_summary_called_in_run(self):
        src = _voice_loop_src()
        assert "tracker.log_summary" in src


# ── Baseline document ─────────────────────────────────────────────────────────


class TestBaselineDocument:
    def test_baseline_doc_exists(self):
        assert BASELINE_DOC.exists()

    def test_doc_has_voice_pipeline_section(self):
        doc = _baseline_doc()
        assert "Voice Pipeline" in doc or "voice pipeline" in doc.lower()

    def test_doc_has_10_sample_interactions(self):
        doc = _baseline_doc()
        # The table should have 10 data rows
        count = sum(
            1
            for line in doc.splitlines()
            if line.startswith("| ") and any(c.isdigit() for c in line[:5])
        )
        assert count >= 10

    def test_doc_has_stt_stage(self):
        assert "stt" in _baseline_doc().lower() or "STT" in _baseline_doc()

    def test_doc_has_llm_stage(self):
        assert "LLM" in _baseline_doc() or "llm" in _baseline_doc()

    def test_doc_has_tts_stage(self):
        assert "TTS" in _baseline_doc() or "tts" in _baseline_doc()

    def test_doc_identifies_dominant_stage(self):
        doc = _baseline_doc()
        # Should explicitly state which stage is the dominant latency contributor
        assert "dominant" in doc.lower() or "largest" in doc.lower() or "majority" in doc.lower()

    def test_doc_has_total_column(self):
        assert "total_s" in _baseline_doc() or "total" in _baseline_doc().lower()
