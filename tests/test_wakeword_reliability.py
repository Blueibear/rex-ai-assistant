"""US-046 wake-word reliability and streaming-frame regression tests."""

from __future__ import annotations

import json
import statistics
import wave
from importlib.metadata import version
from pathlib import Path

import numpy as np

from rex.wakeword.utils import evaluate_wakeword, load_wakeword_model_with_metadata


def test_openwakeword_long_frame_aggregates_native_stream_chunks() -> None:
    class StreamingModel:
        _rex_stream_chunk_samples = 4

        def __init__(self) -> None:
            self.calls = 0

        def predict(self, _frame: np.ndarray) -> dict[str, float]:
            self.calls += 1
            return {"hey jarvis": 0.95 if self.calls == 2 else 0.01}

    model = StreamingModel()
    result = evaluate_wakeword(model, np.ones(12, dtype=np.int16), threshold=0.5)

    assert result.triggered is True
    assert result.confidence == 0.95
    assert model.calls == 3


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "wakeword"
REPORT_PATH = Path(__file__).parents[1] / "docs" / "voice" / "wakeword-report.md"
SAMPLE_RATE_HZ = 16_000
DETECTION_WINDOW_SAMPLES = SAMPLE_RATE_HZ
MODEL_THRESHOLD = 0.5
PROMOTION_THRESHOLD = 0.9


def _load_fixture_audio(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        source_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if source_rate == SAMPLE_RATE_HZ:
        return audio
    target_length = max(round(audio.size * SAMPLE_RATE_HZ / source_rate), 1)
    source_positions = np.arange(audio.size, dtype=np.float64)
    target_positions = np.linspace(0, max(audio.size - 1, 0), target_length)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _evaluate_fixture() -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))
    model, selection = load_wakeword_model_with_metadata(
        keyword="hey jarvis",
        backend="openwakeword",
        fallback_keyword="hey jarvis",
    )
    rows: list[dict[str, object]] = []
    for sample in manifest["samples"]:
        reset = getattr(model, "reset", None)
        if callable(reset):
            reset()
        audio = _load_fixture_audio(FIXTURE_DIR / sample["file"])
        detected = False
        first_detection_ms: int | None = None
        for offset in range(0, int(audio.size), DETECTION_WINDOW_SAMPLES):
            frame = audio[offset : offset + DETECTION_WINDOW_SAMPLES]
            if frame.size < DETECTION_WINDOW_SAMPLES:
                frame = np.pad(frame, (0, DETECTION_WINDOW_SAMPLES - int(frame.size)))
            result = evaluate_wakeword(model, frame, threshold=MODEL_THRESHOLD)
            if result.triggered and not detected:
                detected = True
                first_detection_ms = ((offset // DETECTION_WINDOW_SAMPLES) + 1) * 1000
        rows.append(
            {
                "file": sample["file"],
                "label": sample["label"],
                "phrase": sample["phrase"],
                "detected": detected,
                "first_detection_ms": first_detection_ms,
            }
        )

    tp = sum(row["label"] == "positive" and row["detected"] for row in rows)
    fn = sum(row["label"] == "positive" and not row["detected"] for row in rows)
    fp = sum(row["label"] == "negative" and row["detected"] for row in rows)
    tn = sum(row["label"] == "negative" and not row["detected"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    latencies = [
        int(row["first_detection_ms"])
        for row in rows
        if row["label"] == "positive" and row["first_detection_ms"] is not None
    ]
    metrics: dict[str, object] = {
        "active_label": selection.active_label,
        "detector_version": version("openwakeword"),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "median_detection_ms": int(statistics.median(latencies)) if latencies else None,
    }
    return metrics, rows


def _render_report(metrics: dict[str, object], rows: list[dict[str, object]]) -> str:
    precision = float(metrics["precision"])
    recall = float(metrics["recall"])
    passed = precision >= PROMOTION_THRESHOLD and recall >= PROMOTION_THRESHOLD
    latency = metrics["median_detection_ms"]
    lines = [
        "# Wake-word Reliability Report",
        "",
        "This report is generated deterministically by `tests/test_wakeword_reliability.py` from the tracked synthetic acoustic fixture in `tests/fixtures/wakeword/`.",
        "",
        "## Controlled result",
        "",
        f"- Active model: built-in openWakeWord `{metrics['active_label']}`",
        f"- Detector package: `openwakeword {metrics['detector_version']}`",
        f"- Model activation threshold: `{MODEL_THRESHOLD:.2f}`",
        f"- Promotion threshold: precision >= `{PROMOTION_THRESHOLD:.2f}` and recall >= `{PROMOTION_THRESHOLD:.2f}`",
        f"- Confusion matrix: TP `{metrics['tp']}`, FN `{metrics['fn']}`, FP `{metrics['fp']}`, TN `{metrics['tn']}`",
        f"- Precision: **{precision:.3f}**",
        f"- Recall: **{recall:.3f}**",
        f"- Median positive detection latency: **{latency if latency is not None else 'N/A'} ms** (end of the first 1-second Rex detection window that contains an accepted native openWakeWord frame)",
        f"- Threshold result: **{'PASS' if passed else 'FAIL'}**",
        (
            "- Product classification: **beta**"
            if not passed
            else "- Product classification: eligible for review; no automatic promotion"
        ),
        "",
        "The fixture is intentionally small and synthetic. Passing it would be necessary but not sufficient for a production wake-word claim; broader microphones, speakers, distances, accents, room noise, and continuous negative audio still require deployment evidence.",
        "",
        "## Samples",
        "",
        "| File | Expected | Phrase | Detected | First detection window end |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        detection = "yes" if row["detected"] else "no"
        latency_text = (
            f"{row['first_detection_ms']} ms" if row["first_detection_ms"] is not None else "?"
        )
        lines.append(
            f"| `{row['file']}` | {row['label']} | `{row['phrase']}` | {detection} | {latency_text} |"
        )
    lines.extend(["", "## Decision", ""])
    if passed:
        lines.append(
            "The controlled threshold passes, but wake-word is not automatically promoted; broader deployment evidence is still required before changing the surface classification."
        )
    else:
        lines.append(
            "The controlled threshold fails, so wake-word remains **beta** and is not part of the release-verified voice contract."
        )
    return "\n".join(lines) + "\n"


def test_controlled_fixture_writes_tracked_reliability_report() -> None:
    metrics, rows = _evaluate_fixture()
    report = _render_report(metrics, rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")

    assert metrics["tp"] == 4
    assert metrics["fn"] == 0
    assert metrics["fp"] == 1
    assert metrics["tn"] == 7
    assert metrics["precision"] == 0.8
    assert metrics["recall"] == 1.0
    assert int(metrics["median_detection_ms"]) > 0
    assert "Product classification: **beta**" in report

    repo_root = Path(__file__).parents[1]
    for relative_path in ("README.md", "SURFACE-CLASSIFICATION.md"):
        content = (repo_root / relative_path).read_text(encoding="utf-8").lower()
        assert "wake" in content and "beta" in content
