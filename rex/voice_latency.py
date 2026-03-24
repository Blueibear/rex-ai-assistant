"""Voice pipeline latency tracking.

Records per-stage timing for each voice interaction and logs a summary.
Stages tracked:
  - stt_start / stt_end          — Whisper transcription
  - llm_start / llm_end          — LLM first token (approx) and full response
  - tts_synthesis_start          — TTS engine begins synthesis
  - tts_first_chunk              — First audio chunk produced (XTTS path)
  - playback_start               — Audio sent to output device

Usage::

    from rex.voice_latency import VoiceLatencyTracker

    tracker = VoiceLatencyTracker()
    tracker.mark("stt_start")
    transcript = await transcribe(audio)
    tracker.mark("stt_end")
    ...
    latency = tracker.summary()
    logger.info("Voice latency: %s", latency)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VoiceLatencyTracker:
    """Tracks wall-clock timestamps for each stage of one voice interaction."""

    _marks: dict[str, float] = field(default_factory=dict)

    def mark(self, stage: str) -> float:
        """Record the current time for *stage*. Returns the timestamp."""
        t = time.perf_counter()
        self._marks[stage] = t
        return t

    def elapsed(self, start_stage: str, end_stage: str) -> float | None:
        """Return elapsed seconds between two stages, or None if either is missing."""
        t0 = self._marks.get(start_stage)
        t1 = self._marks.get(end_stage)
        if t0 is None or t1 is None:
            return None
        return round(t1 - t0, 3)

    def summary(self) -> dict[str, float | None]:
        """Return a dict of labelled durations (seconds) for the interaction."""
        return {
            "stt_s": self.elapsed("stt_start", "stt_end"),
            "llm_s": self.elapsed("llm_start", "llm_end"),
            "tts_synthesis_s": self.elapsed("tts_synthesis_start", "tts_synthesis_end"),
            "tts_first_chunk_s": self.elapsed("tts_synthesis_start", "tts_first_chunk"),
            "playback_start_s": self.elapsed("stt_end", "playback_start"),
            "total_s": self.elapsed("stt_start", "playback_start"),
        }

    def log_summary(self) -> None:
        """Log a one-line summary of interaction latencies at INFO level."""
        s = self.summary()
        parts = [f"{k}={v:.3f}" for k, v in s.items() if v is not None]
        logger.info("[latency] %s", "  ".join(parts))
