"""Music Assistant HTTP client for Rex.

Sends playback commands to a running Music Assistant instance
(https://music-assistant.io/) via its REST API.

If ``music_assistant_url`` is not set in ``AppConfig``, all methods raise
``IntegrationNotConfiguredError`` so the rest of the system degrades
gracefully.
"""

from __future__ import annotations

import logging

from rex.assistant_errors import IntegrationNotConfiguredError

logger = logging.getLogger(__name__)


class MusicAssistantClient:
    """HTTP client for Music Assistant playback control.

    Args:
        base_url: Base URL of the Music Assistant server (e.g. ``http://localhost:8095``).
        token: Optional bearer token for authenticated Music Assistant instances.

    If ``base_url`` is falsy the client starts in "not configured" mode and all
    playback methods raise :class:`IntegrationNotConfiguredError`.
    """

    def __init__(self, base_url: str | None = None, token: str | None = None) -> None:
        self._base_url = (base_url or "").rstrip("/")
        self._token = token

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_configured(self) -> None:
        if not self._base_url:
            raise IntegrationNotConfiguredError(
                "Music Assistant: not configured (set music_assistant_url)"
            )

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _post(self, path: str, payload: dict) -> dict:
        """POST *payload* to *path* and return the parsed JSON response."""
        try:
            import json
            import urllib.error
            import urllib.request

            url = f"{self._base_url}{path}"
            data = json.dumps(payload).encode()
            req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read().decode()
                return json.loads(body) if body.strip() else {}
        except urllib.error.HTTPError as exc:
            logger.error("Music Assistant HTTP error %s for %s", exc.code, path)
            raise
        except Exception as exc:
            logger.error("Music Assistant request failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def play(self, query: str, room: str | None = None) -> dict:
        """Start playback of *query* (artist, track, playlist, etc.).

        Args:
            query: Search string passed to Music Assistant.
            room: Optional room/player name to target.  ``None`` targets
                the default player.

        Returns:
            Response dict from Music Assistant.

        Raises:
            IntegrationNotConfiguredError: If the client is not configured.
        """
        self._check_configured()
        payload: dict = {"query": query}
        if room:
            payload["player_id"] = room
        logger.info("Music Assistant: play %r room=%r", query, room, extra={"event": "ma_play"})
        return self._post("/api/players/play_media", payload)

    def pause(self, room: str | None = None) -> dict:
        """Pause the active player.

        Args:
            room: Optional room/player name.  ``None`` targets the default player.

        Returns:
            Response dict from Music Assistant.

        Raises:
            IntegrationNotConfiguredError: If the client is not configured.
        """
        self._check_configured()
        payload: dict = {}
        if room:
            payload["player_id"] = room
        logger.info("Music Assistant: pause room=%r", room, extra={"event": "ma_pause"})
        return self._post("/api/players/pause", payload)

    def resume(self, room: str | None = None) -> dict:
        """Resume a paused player.

        Args:
            room: Optional room/player name.  ``None`` targets the default player.

        Returns:
            Response dict from Music Assistant.

        Raises:
            IntegrationNotConfiguredError: If the client is not configured.
        """
        self._check_configured()
        payload: dict = {}
        if room:
            payload["player_id"] = room
        logger.info("Music Assistant: resume room=%r", room, extra={"event": "ma_resume"})
        return self._post("/api/players/resume", payload)

    def skip(self, room: str | None = None) -> dict:
        """Skip to the next track.

        Args:
            room: Optional room/player name.  ``None`` targets the default player.

        Returns:
            Response dict from Music Assistant.

        Raises:
            IntegrationNotConfiguredError: If the client is not configured.
        """
        self._check_configured()
        payload: dict = {}
        if room:
            payload["player_id"] = room
        logger.info("Music Assistant: skip room=%r", room, extra={"event": "ma_skip"})
        return self._post("/api/players/next", payload)

    def set_volume(self, level: int, room: str | None = None) -> dict:
        """Set the volume of a player.

        Args:
            level: Volume level 0–100.
            room: Optional room/player name.  ``None`` targets the default player.

        Returns:
            Response dict from Music Assistant.

        Raises:
            IntegrationNotConfiguredError: If the client is not configured.
            ValueError: If *level* is outside the 0–100 range.
        """
        self._check_configured()
        if not 0 <= level <= 100:
            raise ValueError(f"Volume level must be 0–100, got {level}")
        payload: dict = {"volume_level": level}
        if room:
            payload["player_id"] = room
        logger.info(
            "Music Assistant: set_volume %d room=%r", level, room, extra={"event": "ma_volume"}
        )
        return self._post("/api/players/volume_set", payload)
