"""Async MQTT client wrapper used by Rex voice nodes.

The implementation is tailored for TLS-enabled brokers with automatic
reconnection, payload validation, and a watchdog heartbeat.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import ssl
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from rex.assistant_errors import ConfigurationError
from rex.config import settings

try:
    from asyncio_mqtt import Client, MqttError, Message
except ImportError as exc:  # pragma: no cover - handled during runtime
    raise ImportError(
        "asyncio-mqtt is required for Rex MQTT features. "
        "Install via `pip install asyncio-mqtt`."
    ) from exc

logger = logging.getLogger(__name__)

MessageHandler = Callable[[str, Dict[str, Any]], Awaitable[None] | None]


class PayloadValidationError(RuntimeError):
    """Raised when a payload does not meet the expected schema."""


def _topic_matches(subscription: str, topic: str) -> bool:
    """Return True if an MQTT topic matches a subscription filter."""
    sub_levels = subscription.split("/")
    topic_levels = topic.split("/")

    for idx, level in enumerate(sub_levels):
        if level == "#":
            return True
        if idx >= len(topic_levels):
            return False
        if level in {"+", topic_levels[idx]}:
            continue
        return False

    return len(topic_levels) == len(sub_levels)


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(tz=timezone.utc).isoformat()


def _resolve_optional_path(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    resolved = Path(path).expanduser()
    return resolved if resolved.exists() else None


@dataclass
class Subscription:
    topic: str
    qos: int = 1
    handlers: List[MessageHandler] = field(default_factory=list)


class RexMQTTClient:
    """High-level MQTT client with TLS, watchdog, and payload validation."""

    DEFAULT_SUBSCRIPTIONS: Tuple[str, ...] = ("rex/audio_in", "rex/audio_out", "rex/ha_cmd")

    def __init__(
        self,
        *,
        broker: Optional[str] = None,
        port: Optional[int] = None,
        tls_enabled: Optional[bool] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: Optional[str] = None,
        keepalive: Optional[int] = None,
        tls_ca: Optional[str] = None,
        tls_cert: Optional[str] = None,
        tls_key: Optional[str] = None,
        tls_insecure: Optional[bool] = None,
        watchdog_interval: Optional[int] = None,
        watchdog_timeout: Optional[int] = None,
        node_id: Optional[str] = None,
        health_topic: str = "rex/health/core",
        required_fields: Iterable[str] = ("node_id", "timestamp"),
        reconnect_interval: float = 5.0,
        reconnect_interval_max: float = 60.0,
    ) -> None:
        cfg = settings
        self._broker = broker or cfg.mqtt_broker
        self._port = int(port or cfg.mqtt_port)
        self._tls_enabled = bool(cfg.mqtt_tls if tls_enabled is None else tls_enabled)
        self._username = username if username is not None else cfg.mqtt_username
        self._password = password if password is not None else cfg.mqtt_password
        self._client_id = (client_id or cfg.mqtt_client_id or f"rex-core-{uuid4().hex[:8]}").strip()
        self._keepalive = int(keepalive or cfg.mqtt_keepalive)
        self._tls_ca = tls_ca if tls_ca is not None else cfg.mqtt_tls_ca
        self._tls_cert = tls_cert if tls_cert is not None else cfg.mqtt_tls_cert
        self._tls_key = tls_key if tls_key is not None else cfg.mqtt_tls_key
        self._tls_insecure = bool(cfg.mqtt_tls_insecure if tls_insecure is None else tls_insecure)
        self._watchdog_interval = int(watchdog_interval or cfg.mqtt_watchdog_interval)
        self._watchdog_timeout = int(watchdog_timeout or cfg.mqtt_watchdog_timeout)
        self._node_id = node_id or cfg.mqtt_node_id or "rex_core"
        self._health_topic = health_topic
        self._required_fields = tuple(required_fields)
        self._reconnect_interval = reconnect_interval
        self._reconnect_interval_max = reconnect_interval_max

        self._subscriptions: Dict[str, Subscription] = {
            topic: Subscription(topic=topic) for topic in self.DEFAULT_SUBSCRIPTIONS
        }

        self._client: Optional[Client] = None
        self._connection_task: Optional[asyncio.Task[None]] = None
        self._watchdog_task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self._connected_event = asyncio.Event()
        self._last_message_at = time.monotonic()
        if not self._broker:
            raise ConfigurationError("MQTT broker host is required (set REX_MQTT_BROKER).")

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    @property
    def is_connected(self) -> bool:
        return self._connected_event.is_set()

    async def start(self, *, wait_connected: bool = True, timeout: Optional[float] = 20.0) -> None:
        """Start the MQTT connection loop and optional watchdog."""
        if self._connection_task is None or self._connection_task.done():
            self._stop_event.clear()
            self._connection_task = asyncio.create_task(
                self._connection_loop(), name="rex-mqtt-connection"
            )
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(
                self._watchdog_loop(), name="rex-mqtt-watchdog"
            )

        if wait_connected:
            await self.wait_connected(timeout=timeout)

    async def stop(self) -> None:
        """Stop connection loop and close the MQTT client."""
        self._stop_event.set()

        for task in (self._watchdog_task, self._connection_task):
            if task is None:
                continue
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        self._watchdog_task = None
        self._connection_task = None

        client = self._client
        self._client = None
        if client is not None:
            with suppress(MqttError):
                await client.disconnect()

        self._connected_event.clear()

    async def wait_connected(self, *, timeout: Optional[float] = None) -> None:
        """Block until the client is connected or timeout expires."""
        if timeout is None:
            await self._connected_event.wait()
            return
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("MQTT connection timed out.") from exc

    async def publish(
        self,
        topic: str,
        payload: Dict[str, Any] | str | bytes,
        *,
        qos: int = 1,
        retain: bool = False,
        ensure_base_fields: bool = True,
    ) -> None:
        """Publish a message to the broker."""
        client = self._client
        if client is None or not self.is_connected:
            raise RuntimeError("MQTT client is not connected.")

        if isinstance(payload, dict):
            message = dict(payload)
            if ensure_base_fields:
                message.setdefault("node_id", self._node_id)
                message.setdefault("timestamp", _now_iso())
            self._validate_payload(message, context=f"publish:{topic}", check_timestamp=False)
            try:
                payload_bytes = json.dumps(message, separators=(",", ":")).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise PayloadValidationError(
                    f"[publish:{topic}] Payload is not JSON serializable."
                ) from exc
        elif isinstance(payload, str):
            payload_bytes = payload.encode("utf-8")
        else:
            payload_bytes = payload

        try:
            await client.publish(topic, payload_bytes, qos=qos, retain=retain)
        except MqttError:
            self._connected_event.clear()
            raise

    async def add_subscription(
        self,
        topic: str,
        handler: MessageHandler,
        *,
        qos: int = 1,
        immediate_sync: bool = True,
    ) -> None:
        """Register a handler for a topic filter and ensure subscription."""
        subscription = self._subscriptions.get(topic)
        if subscription is None:
            subscription = Subscription(topic=topic, qos=qos)
            self._subscriptions[topic] = subscription
        else:
            subscription.qos = max(subscription.qos, qos)
        subscription.handlers.append(handler)

        if immediate_sync and self._client is not None and self.is_connected:
            await self._client.subscribe(topic, qos=subscription.qos)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    async def _connection_loop(self) -> None:
        """Maintain the MQTT connection with exponential backoff."""
        backoff = self._reconnect_interval
        while not self._stop_event.is_set():
            try:
                tls_context = self._build_tls_context() if self._tls_enabled else None
                async with Client(
                    hostname=self._broker,
                    port=self._port,
                    username=self._username,
                    password=self._password,
                    client_id=self._client_id,
                    keepalive=self._keepalive,
                    tls_context=tls_context,
                ) as active_client:
                    self._client = active_client
                    await self._apply_subscriptions(active_client)
                    self._connected_event.set()
                    self._last_message_at = time.monotonic()
                    backoff = self._reconnect_interval

                    async with active_client.unfiltered_messages() as messages:
                        await self._apply_subscriptions(active_client)
                        async for message in messages:
                            self._last_message_at = time.monotonic()
                            await self._dispatch_message(message)

            except MqttError as exc:
                logger.warning("MQTT connection lost: %s", exc)
                self._connected_event.clear()
                self._client = None
                if self._stop_event.is_set():
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._reconnect_interval_max)
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Unexpected error in MQTT connection loop.")
                self._connected_event.clear()
                self._client = None
                if self._stop_event.is_set():
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._reconnect_interval_max)
            else:
                # Connection closed gracefully; wait before reconnecting unless stopping.
                self._connected_event.clear()
                self._client = None
                if self._stop_event.is_set():
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._reconnect_interval_max)

    async def _dispatch_message(self, message: Message) -> None:
        """Dispatch incoming messages to registered handlers."""
        payload_bytes = message.payload
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Discarding non-JSON payload on %s: %s", message.topic, exc)
            return

        try:
            self._validate_payload(payload, context=message.topic)
        except PayloadValidationError as exc:
            logger.warning("Invalid payload on %s: %s", message.topic, exc)
            return

        matching_handlers = [
            handler
            for subscription in self._subscriptions.values()
            if _topic_matches(subscription.topic, message.topic)
            for handler in subscription.handlers
        ]

        if not matching_handlers:
            logger.debug("No handler registered for topic %s", message.topic)
            return

        for handler in matching_handlers:
            try:
                result = handler(message.topic, payload)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Handler error for topic %s", message.topic)

    async def _apply_subscriptions(self, client: Client) -> None:
        """Ensure all known subscriptions are active."""
        if not self._subscriptions:
            return
        subscriptions = [(sub.topic, sub.qos) for sub in self._subscriptions.values()]
        for topic, qos in subscriptions:
            await client.subscribe(topic, qos=qos)

    async def _watchdog_loop(self) -> None:
        """Periodically publish a heartbeat if traffic is silent."""
        await asyncio.sleep(self._watchdog_interval)
        while not self._stop_event.is_set():
            if self.is_connected:
                idle_seconds = time.monotonic() - self._last_message_at
                if idle_seconds >= self._watchdog_timeout:
                    heartbeat = {
                        "node_id": self._node_id,
                        "timestamp": _now_iso(),
                        "type": "watchdog",
                        "idle_seconds": round(idle_seconds, 2),
                    }
                    try:
                        await self.publish(
                            self._health_topic,
                            heartbeat,
                            qos=0,
                            retain=False,
                            ensure_base_fields=False,
                        )
                        self._last_message_at = time.monotonic()
                    except Exception:  # pragma: no cover - defensive logging
                        logger.exception("Failed to publish MQTT watchdog heartbeat.")
            await asyncio.sleep(self._watchdog_interval)

    def _build_tls_context(self) -> Optional[ssl.SSLContext]:
        """Create an SSL context based on configuration."""
        ca_path = _resolve_optional_path(self._tls_ca)
        cert_path = _resolve_optional_path(self._tls_cert)
        key_path = _resolve_optional_path(self._tls_key)

        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        if ca_path:
            context.load_verify_locations(cafile=str(ca_path))
        if cert_path and key_path:
            context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
        if self._tls_insecure:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

        return context

    def _validate_payload(
        self,
        payload: Dict[str, Any],
        *,
        context: str,
        check_timestamp: bool = True,
    ) -> None:
        """Validate that a payload matches the expected schema."""
        error_prefix = f"[{context}] " if context else ""

        if not isinstance(payload, dict):
            raise PayloadValidationError(f"{error_prefix}Payload must be a JSON object.")

        missing = [field for field in self._required_fields if field not in payload]
        if missing:
            raise PayloadValidationError(f"{error_prefix}Missing fields {missing!r}")

        node_id = payload.get("node_id")
        if not isinstance(node_id, str) or not node_id.strip():
            raise PayloadValidationError(f"{error_prefix}node_id must be a non-empty string.")

        timestamp_raw = payload.get("timestamp")
        if not isinstance(timestamp_raw, str):
            raise PayloadValidationError(f"{error_prefix}timestamp must be an ISO-8601 string.")

        if check_timestamp:
            try:
                parsed = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
            except ValueError as exc:  # pragma: no cover - defensive
                raise PayloadValidationError(
                    f"{error_prefix}timestamp must be an ISO-8601 string."
                ) from exc

            now = datetime.now(tz=timezone.utc)
            if parsed > now + timedelta(seconds=30):
                raise PayloadValidationError(f"{error_prefix}timestamp is in the future.")


_default_client: Optional[RexMQTTClient] = None


def get_mqtt_client() -> RexMQTTClient:
    """Return a shared RexMQTTClient instance."""
    global _default_client
    if _default_client is None:
        _default_client = RexMQTTClient()
    return _default_client
