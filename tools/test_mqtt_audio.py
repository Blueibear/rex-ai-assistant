"""CLI helper to verify Rex MQTT audio roundtrip locally.

The tool publishes a test clip (from a WAV file or synthetic tone) to the
`rex/audio_in` topic and waits for a reply on `rex/tts/<node_id>`.
"""

from __future__ import annotations

import argparse
import asyncio
import audioop
import base64
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

from rex.config import settings
from rex.mqtt_client import RexMQTTClient


def _utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _load_audio(audio_path: Optional[Path], sample_rate: int) -> Tuple[np.ndarray, int]:
    if audio_path and audio_path.exists():
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        return np.asarray(data, dtype=np.float32), sr

    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    carrier = 440.0
    data = 0.2 * np.sin(2 * math.pi * carrier * t)
    return data.astype(np.float32), sample_rate


def _encode_mulaw(audio: np.ndarray) -> bytes:
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    return audioop.lin2ulaw(pcm16.tobytes(), 2)


@dataclass
class RoundTripResult:
    transcript: Optional[str]
    reply_text: Optional[str]
    latency_ms: Optional[float]
    response_topic: Optional[str]
    received_at: str


async def run_test(
    *,
    audio_path: Optional[Path],
    node_id: str,
    sample_rate: int,
    timeout: float,
) -> RoundTripResult:
    client = RexMQTTClient()
    result: RoundTripResult = RoundTripResult(
        transcript=None,
        reply_text=None,
        latency_ms=None,
        response_topic=None,
        received_at="",
    )
    event = asyncio.Event()

    async def _handle_reply(topic: str, payload):
        if payload.get("node_id") != node_id:
            return
        result.transcript = payload.get("transcript")
        result.reply_text = payload.get("reply_text")
        result.latency_ms = payload.get("latency_ms")
        result.response_topic = topic
        result.received_at = _utc_iso()
        event.set()

    await client.add_subscription(f"rex/tts/{node_id}", _handle_reply)
    await client.start()

    start_time = time.perf_counter()
    try:
        samples, detected_rate = _load_audio(audio_path, sample_rate)
        sample_rate = detected_rate or sample_rate
        payload_bytes = _encode_mulaw(samples)
        message = {
            "node_id": node_id,
            "timestamp": _utc_iso(),
            "sequence": 1,
            "codec": "mulaw",
            "sample_rate": sample_rate,
            "audio": base64.b64encode(payload_bytes).decode("ascii"),
            "end": True,
        }
        await client.publish("rex/audio_in", message, qos=1)
        await asyncio.wait_for(event.wait(), timeout=timeout)
    finally:
        await client.stop()

    end_time = time.perf_counter()
    if result.latency_ms is None:
        result.latency_ms = round((end_time - start_time) * 1000, 2)
    return result


def _append_result_to_log(result: RoundTripResult, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"[{result.received_at or _utc_iso()}] topic={result.response_topic or 'n/a'}",
        f"  transcript={result.transcript or '<none>'}",
        f"  reply={result.reply_text or '<none>'}",
        f"  latency_ms={result.latency_ms if result.latency_ms is not None else 'n/a'}",
    ]
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a test clip to rex/audio_in and await rex/tts/<node> response.")
    parser.add_argument("--audio", type=Path, help="Path to WAV file to publish (default: synthetic tone)")
    parser.add_argument("--node-id", default=settings.mqtt_node_id, help="Override node identifier (default: config value)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate for synthetic audio (default: 16000)")
    parser.add_argument("--timeout", type=float, default=15.0, help="Seconds to wait for Rex reply (default: 15)")
    parser.add_argument("--log", type=Path, default=Path("logs/mqtt_roundtrip_test.log"), help="Path to append test results")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = asyncio.run(
            run_test(
                audio_path=args.audio,
                node_id=args.node_id,
                sample_rate=args.sample_rate,
                timeout=args.timeout,
            )
        )
    except asyncio.TimeoutError:
        print("Timed out waiting for Rex reply on MQTT.")
        return 1

    print(f"Received reply topic: {result.response_topic}")
    print(f"Transcript: {result.transcript}")
    print(f"Reply: {result.reply_text}")
    print(f"Latency (ms): {result.latency_ms}")

    if args.log:
        _append_result_to_log(result, args.log)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
