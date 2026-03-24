"""Tests for US-171: Immediate audio acknowledgment on wake word (verify and enforce)."""

import asyncio
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
VOICE_LOOP_SRC = REPO_ROOT / "rex" / "voice_loop.py"
PERF_DOC = REPO_ROOT / "docs" / "performance-baseline.md"


def _src() -> str:
    return VOICE_LOOP_SRC.read_text(encoding="utf-8")


# ── Source-level checks ────────────────────────────────────────────────────────


class TestSourceAck:
    def _get_run_body(self) -> str:
        src = _src()
        idx = src.index("async def run(self, max_interactions")
        return src[idx : idx + 4000]

    def test_safe_acknowledge_method_exists(self):
        assert "_safe_acknowledge" in _src()

    def test_safe_acknowledge_is_async(self):
        assert "async def _safe_acknowledge" in _src()

    def test_run_uses_create_task_for_ack(self):
        body = self._get_run_body()
        assert "create_task" in body
        assert "_safe_acknowledge" in body

    def test_run_does_not_await_acknowledge_directly(self):
        """Ack must not be awaited directly (would block recording start)."""
        body = self._get_run_body()
        assert "await self._acknowledge()" not in body

    def test_safe_acknowledge_has_try_except(self):
        src = _src()
        idx = src.index("async def _safe_acknowledge")
        body = src[idx : idx + 400]
        assert "except" in body

    def test_safe_acknowledge_logs_warning_on_error(self):
        src = _src()
        idx = src.index("async def _safe_acknowledge")
        body = src[idx : idx + 400]
        assert "logger.warning" in body

    def test_ack_fires_before_record_phrase(self):
        body = self._get_run_body()
        ack_pos = body.index("_safe_acknowledge")
        record_pos = body.index("record_phrase")
        assert ack_pos < record_pos


# ── Functional tests ───────────────────────────────────────────────────────────


class TestSafeAcknowledge:
    @pytest.mark.asyncio
    async def test_safe_acknowledge_calls_acknowledge(self):
        from rex.voice_loop import VoiceLoop

        called = []

        async def _mock_ack():
            called.append(True)

        loop = VoiceLoop.__new__(VoiceLoop)
        loop._acknowledge = _mock_ack

        await loop._safe_acknowledge()
        assert called == [True]

    @pytest.mark.asyncio
    async def test_safe_acknowledge_does_not_raise_on_failure(self):
        from rex.voice_loop import VoiceLoop

        async def _bad_ack():
            raise RuntimeError("Audio device missing")

        loop = VoiceLoop.__new__(VoiceLoop)
        loop._acknowledge = _bad_ack

        # Should not raise
        await loop._safe_acknowledge()

    @pytest.mark.asyncio
    async def test_safe_acknowledge_noop_when_none(self):
        from rex.voice_loop import VoiceLoop

        loop = VoiceLoop.__new__(VoiceLoop)
        loop._acknowledge = None

        # Should not raise
        await loop._safe_acknowledge()

    @pytest.mark.asyncio
    async def test_ack_runs_concurrently_with_recording(self):
        """Verify ack fires as a task while record_phrase runs."""
        import numpy as np

        from rex.voice_loop import VoiceLoop

        order = []

        async def _mock_ack():
            order.append("ack_start")
            await asyncio.sleep(0.05)
            order.append("ack_done")

        async def _mock_record():
            order.append("record_start")
            await asyncio.sleep(0)
            return np.zeros(1000, dtype=np.float32)

        # Patch a minimal VoiceLoop that fires ack via create_task then records
        loop = VoiceLoop.__new__(VoiceLoop)
        loop._acknowledge = _mock_ack

        ack_task = asyncio.create_task(loop._safe_acknowledge())
        await _mock_record()
        await ack_task

        # record_start should appear before ack_done (i.e., they overlapped)
        assert order.index("record_start") < order.index("ack_done")


# ── Performance documentation ──────────────────────────────────────────────────


class TestPerformanceDoc:
    def test_perf_doc_exists(self):
        assert PERF_DOC.exists()

    def test_perf_doc_has_ack_section(self):
        content = PERF_DOC.read_text(encoding="utf-8")
        assert "acknowledgment" in content.lower() or "Acknowledgment" in content

    def test_perf_doc_mentions_concurrent(self):
        content = PERF_DOC.read_text(encoding="utf-8")
        assert "concurrent" in content.lower() or "create_task" in content
