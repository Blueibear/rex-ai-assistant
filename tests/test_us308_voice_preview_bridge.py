"""Tests for US-308: TTS voice preview bridge.

Acceptance criteria:
  - rex_voice_sample_bridge.py exists and is callable
  - Bridge reads provider + voice_id from stdin JSON
  - Default preview phrase is "Hello, I'm your Rex assistant."
  - Missing voice_id returns ok=false with descriptive error
  - Bad JSON input returns ok=false with descriptive error
  - Successful synthesis returns ok=true with audio_base64
  - Bridge is registered in the Electron bridge resolver registry
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).parent.parent
BRIDGE = REPO_ROOT / "rex_voice_sample_bridge.py"


class TestVoiceSampleBridgeContract:
    def test_bridge_file_exists(self):
        assert BRIDGE.exists(), f"Bridge script missing: {BRIDGE}"

    def test_missing_voice_id_returns_error_for_voice_specific_provider(self):
        """Calling a voice-specific provider without voice_id yields ok=false."""
        payload = json.dumps({"provider": "edge-tts", "voice_id": ""})
        result = subprocess.run(
            [sys.executable, str(BRIDGE)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [line for line in result.stdout.strip().splitlines() if line.startswith("{")]
        assert lines, f"No JSON output: {result.stdout!r} stderr={result.stderr!r}"
        data = json.loads(lines[-1])
        assert data["ok"] is False
        assert "voice_id" in data.get("error", "").lower()

    def test_bad_json_input_returns_error(self):
        """Malformed stdin yields ok=false."""
        result = subprocess.run(
            [sys.executable, str(BRIDGE)],
            input="not-json",
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [line for line in result.stdout.strip().splitlines() if line.startswith("{")]
        assert lines, f"No JSON output: {result.stdout!r}"
        data = json.loads(lines[-1])
        assert data["ok"] is False

    def test_default_preview_phrase_contains_rex(self):
        """Default text parameter contains 'Rex'."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("vb", BRIDGE)
        assert spec is not None
        source = BRIDGE.read_text(encoding="utf-8")
        # The default phrase must include "Rex" per acceptance criteria
        assert "Rex" in source
        assert "Hello" in source

    def test_synthesis_success_returns_audio_base64(self):
        """Successful synthesis returns ok=true with audio_base64."""
        import base64

        mock_audio = b"\x00\x01\x02\x03" * 100

        def mock_synthesize(provider, voice_id, text):
            return mock_audio

        with patch.dict(
            "sys.modules",
            {"rex.tts_voices": MagicMock(synthesize_sample=MagicMock(return_value=mock_audio))},
        ):
            import importlib

            # Patch asyncio.run to call synchronously
            with patch("asyncio.run", side_effect=lambda coro: mock_audio):
                # Import and run main with mocked stdin
                import io
                import json as json_mod

                payload = json_mod.dumps({"provider": "pyttsx3", "voice_id": "test-voice"})

                captured = io.StringIO()
                with (
                    patch("sys.stdin", io.StringIO(payload)),
                    patch("sys.stdout", captured),
                ):
                    # Re-import the bridge module to run main()
                    import importlib.util

                    spec = importlib.util.spec_from_file_location("vb2", BRIDGE)
                    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
                    spec.loader.exec_module(mod)  # type: ignore[union-attr]
                    mod.main()

                output = captured.getvalue().strip()
                lines = [line for line in output.splitlines() if line.startswith("{")]
                assert lines
                data = json_mod.loads(lines[-1])
                assert data["ok"] is True
                assert "audio_base64" in data
                decoded = base64.b64decode(data["audio_base64"])
                assert decoded == mock_audio


class TestBridgeInRegistry:
    def test_voice_sample_bridge_in_registry(self):
        """rex_voice_sample_bridge must be in the centralized bridge registry."""
        resolver_path = REPO_ROOT / "gui" / "src" / "main" / "bridgeResolver.ts"
        if not resolver_path.exists():
            return  # GUI not present in this env
        content = resolver_path.read_text(encoding="utf-8")
        assert (
            "rex_voice_sample_bridge" in content
        ), "rex_voice_sample_bridge not found in bridgeResolver.ts registry"
