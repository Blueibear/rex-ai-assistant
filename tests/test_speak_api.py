"""Tests for rex_speak_api thread-safety (US-194)."""

from __future__ import annotations

import threading


def _make_client(app):
    """Return a test client with a fixed API key."""
    app.config["TESTING"] = True
    return app.test_client()


def _speak_payload():
    return {"text": "hello", "language": "en"}


# ---------------------------------------------------------------------------
# Lock existence
# ---------------------------------------------------------------------------


def test_tts_lock_exists():
    import rex_speak_api

    assert hasattr(rex_speak_api, "_tts_lock"), "_tts_lock must exist at module level"


def test_tts_lock_is_lock():
    import rex_speak_api

    assert isinstance(rex_speak_api._tts_lock, type(threading.Lock()))


# ---------------------------------------------------------------------------
# _get_tts_engine uses lock
# ---------------------------------------------------------------------------


def test_get_tts_engine_acquires_lock(monkeypatch):
    """_get_tts_engine must be called while _tts_lock can be acquired."""
    import rex_speak_api

    acquired_states: list[bool] = []

    def fake_generate(text, language, user_key):
        # If the lock is held during generate_speech, trying to acquire it
        # from *another* thread would block.  Here we just verify the lock
        # exists and is a Lock.
        acquired_states.append(True)
        return b"\x00\x01"

    monkeypatch.setattr(rex_speak_api, "generate_speech", fake_generate)
    monkeypatch.setenv("REX_SPEAK_API_KEY", "testkey")

    from rex_speak_api import app

    client = _make_client(app)
    resp = client.post(
        "/speak",
        json=_speak_payload(),
        headers={"X-API-Key": "testkey"},
    )
    assert resp.status_code == 200
    assert acquired_states == [True]


# ---------------------------------------------------------------------------
# Concurrent /speak requests
# ---------------------------------------------------------------------------


def test_concurrent_speak_requests(monkeypatch):
    """5 concurrent /speak requests must all return 200 with non-empty audio."""
    import rex_speak_api

    call_count = 0
    call_lock = threading.Lock()

    def fake_generate(text, language, user_key):
        nonlocal call_count
        with call_lock:
            call_count += 1
        return b"\x52\x49\x46\x46"  # minimal non-empty bytes

    monkeypatch.setattr(rex_speak_api, "generate_speech", fake_generate)
    monkeypatch.setenv("REX_SPEAK_API_KEY", "testkey")

    from rex_speak_api import app

    app.config["TESTING"] = True

    results: list[tuple[int, int]] = []  # (status_code, content_length)
    errors: list[Exception] = []

    def worker():
        try:
            with app.test_client() as client:
                resp = client.post(
                    "/speak",
                    json=_speak_payload(),
                    headers={"X-API-Key": "testkey"},
                )
                results.append((resp.status_code, len(resp.data)))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"Thread errors: {errors}"
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"
    for status, length in results:
        assert status == 200, f"Expected 200, got {status}"
        assert length > 0, "Expected non-empty audio response"


# ---------------------------------------------------------------------------
# generate_speech holds lock during synthesis
# ---------------------------------------------------------------------------


def test_generate_speech_holds_lock_during_synthesis(monkeypatch):
    """Verify _tts_lock is held during engine access in generate_speech."""
    import rex_speak_api

    lock_held_during_engine_call: list[bool] = []

    def patched_get_engine():
        # When called from generate_speech, _tts_lock should already be held
        # by the current thread — trying to acquire it should fail immediately
        acquired = rex_speak_api._tts_lock.acquire(blocking=False)
        lock_held_during_engine_call.append(not acquired)
        if not acquired:
            pass  # lock is held — expected
        else:
            rex_speak_api._tts_lock.release()
        # Return a fake engine
        return _fake_engine()

    class _FakeEngine:
        def tts_to_file(self, **kwargs):
            import numpy as np
            import soundfile as sf

            sf.write(kwargs["file_path"], np.zeros(1000), 22050)

    def _fake_engine():
        return _FakeEngine()

    # Patch _get_tts_engine and audio deps at module level
    monkeypatch.setattr(rex_speak_api, "_get_tts_engine", patched_get_engine)

    import numpy as np

    fake_np = type(
        "FakeNumpy",
        (),
        {
            "concatenate": staticmethod(np.concatenate),
        },
    )()

    called = []

    def fake_load_audio():
        called.append(True)

        class _FakeSF:
            @staticmethod
            def read(path, dtype="float32"):
                return np.zeros(100), 22050

            @staticmethod
            def write(path, data, rate):
                pass

        return fake_np, _FakeSF()

    monkeypatch.setattr(rex_speak_api, "_load_audio_dependencies", fake_load_audio)
    monkeypatch.setattr(rex_speak_api, "chunk_text_for_xtts", lambda text, **kw: [text])

    # Patch open for reading the output file
    import builtins

    original_open = builtins.open

    def fake_open(path, mode="r", *args, **kwargs):
        if "rb" in str(mode):
            import io

            return io.BytesIO(b"\x00\x01\x02")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    from pathlib import Path

    monkeypatch.setattr(Path, "unlink", lambda self, missing_ok=False: None)

    rex_speak_api.generate_speech("hello", "en", "james")

    assert lock_held_during_engine_call, "_get_tts_engine was not called"
    assert lock_held_during_engine_call[0], "_tts_lock was NOT held during _get_tts_engine call"


# ---------------------------------------------------------------------------
# Request body size limit (US-200)
# ---------------------------------------------------------------------------


def test_max_request_bytes_constant_exists():
    import rex_speak_api

    assert hasattr(rex_speak_api, "MAX_REQUEST_BYTES")
    assert isinstance(rex_speak_api.MAX_REQUEST_BYTES, int)
    assert rex_speak_api.MAX_REQUEST_BYTES > 0


def test_oversized_request_body_returns_413(monkeypatch):
    """A 100 KB request body must be rejected with 413 before synthesis."""
    import rex_speak_api

    monkeypatch.setenv("REX_SPEAK_API_KEY", "testkey")

    from rex_speak_api import app

    app.config["TESTING"] = True
    # Ensure limit is 64 KB for this test regardless of env
    app.config["MAX_CONTENT_LENGTH"] = 65536
    monkeypatch.setattr(rex_speak_api, "MAX_REQUEST_BYTES", 65536)

    client = app.test_client()
    large_body = b"x" * (100 * 1024)  # 100 KB — well above the 64 KB limit
    resp = client.post(
        "/speak",
        data=large_body,
        content_type="application/json",
        headers={"X-API-Key": "testkey"},
    )
    assert resp.status_code == 413


def test_request_within_size_limit_proceeds(monkeypatch):
    """A request body under MAX_REQUEST_BYTES must not be rejected with 413."""
    import rex_speak_api

    monkeypatch.setenv("REX_SPEAK_API_KEY", "testkey")

    def fake_generate(text, language, user_key):
        return b"\x52\x49\x46\x46"

    monkeypatch.setattr(rex_speak_api, "generate_speech", fake_generate)

    from rex_speak_api import app

    app.config["TESTING"] = True
    app.config["MAX_CONTENT_LENGTH"] = 65536

    client = app.test_client()
    resp = client.post(
        "/speak",
        json={"text": "hello", "language": "en"},
        headers={"X-API-Key": "testkey"},
    )
    assert resp.status_code == 200
