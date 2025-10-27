"""MQTT audio routing pipeline for remote Rex voice nodes."""

from __future__ import annotations

import asyncio
import audioop
import base64
import contextlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

from rex.assistant import Assistant
from rex.assistant_errors import SpeechToTextError, TextToSpeechError
from rex.config import settings
from rex.mqtt_client import RexMQTTClient
from rex.voice_loop import (
    SpeechToText,
    TextToSpeech,
    _resolve_voice_reference,
)

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _decode_mulaw(payload: str) -> bytes:
    return base64.b64decode(payload)


def _encode_mulaw(pcm_bytes: bytes) -> str:
    return base64.b64encode(pcm_bytes).decode("ascii")


@dataclass
class AudioSession:
    node_id: str
    sample_rate: int = 16000
    codec: str = "mulaw"
    created_at: float = field(default_factory=time.perf_counter)
    sequence: int = 0
    buffer: bytearray = field(default_factory=bytearray)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def append_chunk(self, chunk: bytes, sequence: Optional[int]) -> None:
        if sequence is not None and sequence <= self.sequence:
            logger.warning(
                "Out-of-order chunk for node %s: got %s after %s",
                self.node_id,
                sequence,
                self.sequence,
            )
        if sequence is not None:
            self.sequence = sequence
        self.buffer.extend(chunk)

    def reset(self) -> None:
        self.buffer.clear()
        self.sequence = 0
        self.metadata.clear()
        self.created_at = time.perf_counter()


class MqttAudioRouter:
    """Handle audio messages from MQTT nodes and route responses."""

    def __init__(
        self,
        *,
        assistant: Assistant,
        mqtt_client: Optional[RexMQTTClient] = None,
        speech_to_text: Optional[SpeechToText] = None,
        text_to_speech: Optional[TextToSpeech] = None,
    ) -> None:
        self._assistant = assistant
        self._client = mqtt_client or RexMQTTClient()
        self._speech_to_text = speech_to_text
        self._text_to_speech = text_to_speech
        self._voice_reference: Optional[str] = None
        self._sessions: Dict[str, AudioSession] = {}
        self._pending_tasks: set[asyncio.Task[Any]] = set()
        self._started = False

    async def start(self) -> None:
        if self._started:
            return

        await self._ensure_components()
        await self._client.add_subscription("rex/audio_in", self._handle_audio_message)
        await self._client.start(wait_connected=True)
        self._started = True
        logger.info("[MQTT] Audio router started.")

    async def stop(self) -> None:
        if not self._started:
            return

        for task in list(self._pending_tasks):
            task.cancel()
        for task in list(self._pending_tasks):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._pending_tasks.clear()

        await self._client.stop()
        self._started = False
        logger.info("[MQTT] Audio router stopped.")

    async def _ensure_components(self) -> None:
        if self._speech_to_text is None:
            try:
                self._speech_to_text = SpeechToText(
                    settings.whisper_model,
                    settings.whisper_device,
                )
            except SpeechToTextError as exc:
                logger.error("[MQTT] Speech-to-text initialisation failed: %s", exc)
                raise

        if self._voice_reference is None:
            self._voice_reference = _resolve_voice_reference()

        if self._text_to_speech is None:
            self._text_to_speech = TextToSpeech(
                language=settings.speak_language,
                default_speaker=self._voice_reference,
            )

    async def _handle_audio_message(self, topic: str, payload: Dict[str, Any]) -> None:
        node_id = payload.get("node_id")
        if not node_id:
            logger.warning("[MQTT] audio message missing node_id.")
            return

        session = self._sessions.setdefault(node_id, AudioSession(node_id=node_id))

        sample_rate = payload.get("sample_rate")
        if isinstance(sample_rate, int) and sample_rate > 0:
            session.sample_rate = sample_rate

        codec = payload.get("codec") or payload.get("format")
        if codec:
            session.codec = codec

        sequence = payload.get("sequence")
        audio_b64 = payload.get("audio")
        if not audio_b64:
            logger.warning("[MQTT] audio payload missing 'audio' field for node %s.", node_id)
            return

        try:
            chunk = _decode_mulaw(audio_b64)
        except (ValueError, TypeError) as exc:
            logger.warning("[MQTT] Failed to decode audio chunk from node %s: %s", node_id, exc)
            return

        session.append_chunk(chunk, sequence if isinstance(sequence, int) else None)

        if payload.get("end", False) or payload.get("is_final", False):
            task = asyncio.create_task(
                self._process_session(node_id, session, payload),
                name=f"rex-mqtt-audio-{node_id}",
            )
            task.add_done_callback(self._pending_tasks.discard)
            self._pending_tasks.add(task)
            self._sessions.pop(node_id, None)

    async def _process_session(
        self,
        node_id: str,
        session: AudioSession,
        final_payload: Dict[str, Any],
    ) -> None:
        start = session.created_at
        duration_ms = lambda: round((time.perf_counter() - start) * 1000, 2)

        if self._speech_to_text is None or self._text_to_speech is None:
            logger.error("[MQTT] Cannot process session; STT or TTS unavailable.")
            return

        if session.codec.lower() != "mulaw":
            logger.warning("[MQTT] Unsupported codec '%s' from node %s", session.codec, node_id)
            return

        logger.info("[MQTT] Processing audio session for node %s", node_id)

        try:
            pcm16 = audioop.ulaw2lin(bytes(session.buffer), 2)
            audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as exc:
            logger.exception("[MQTT] Failed to decode mu-law audio from node %s: %s", node_id, exc)
            return

        transcript: Optional[str] = None
        try:
            transcript = await self._speech_to_text.transcribe(audio, session.sample_rate)
            logger.info("[MQTT] Node %s transcript: %s", node_id, transcript)
        except Exception as exc:
            logger.exception("[MQTT] Speech-to-text failed for node %s: %s", node_id, exc)
            await self._publish_error(node_id, "stt_error", str(exc), duration_ms())
            return

        try:
            reply = await self._assistant.generate_reply(transcript)
            logger.info("[MQTT] Node %s reply: %s", node_id, reply)
        except Exception as exc:
            logger.exception("[MQTT] LLM generation failed for node %s: %s", node_id, exc)
            await self._publish_error(node_id, "llm_error", str(exc), duration_ms())
            return

        try:
            tts_audio = await self._text_to_speech.synthesise(
                reply,
                speaker_wav=self._voice_reference,
            )
        except TextToSpeechError as exc:
            logger.exception("[MQTT] Text-to-speech failed for node %s: %s", node_id, exc)
            await self._publish_error(node_id, "tts_error", str(exc), duration_ms())
            return

        compressed = audioop.lin2ulaw(tts_audio.to_int16().tobytes(), 2)
        payload = {
            "node_id": node_id,
            "timestamp": _utc_iso(),
            "format": "mulaw",
            "sample_rate": tts_audio.sample_rate,
            "audio": _encode_mulaw(compressed),
            "transcript": transcript,
            "reply_text": reply,
            "latency_ms": duration_ms(),
        }

        await self._client.publish(f"rex/tts/{node_id}", payload, qos=1)
        await self._client.publish(
            "rex/audio_out",
            {
                "node_id": node_id,
                "timestamp": payload["timestamp"],
                "type": "reply",
                "transcript": transcript,
                "reply_text": reply,
                "latency_ms": payload["latency_ms"],
            },
            qos=1,
        )

    async def _publish_error(
        self,
        node_id: str,
        error_type: str,
        message: str,
        latency_ms: float,
    ) -> None:
        payload = {
            "node_id": node_id,
            "timestamp": _utc_iso(),
            "type": "error",
            "error": error_type,
            "message": message,
            "latency_ms": latency_ms,
        }
        await self._client.publish("rex/audio_out", payload, qos=1)
