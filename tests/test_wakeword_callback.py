import asyncio

import pytest
import voice_loop

np = pytest.importorskip("numpy")


def test_wakeword_callback_sets_event_only_on_true(monkeypatch):
    async def _exercise():
        loop = asyncio.get_running_loop()
        listener = voice_loop.WakeWordListener(
            model=object(),
            threshold=0.5,
            sample_rate=16000,
            block_size=1600,
            device=None,
            loop=loop,
        )

        monkeypatch.setattr(voice_loop, "detect_wakeword", lambda *_, **__: False)
        listener._callback(np.zeros((1600, 1), dtype=np.float32), 1600, None, None)
        await asyncio.sleep(0)
        assert not listener._event.is_set()

        monkeypatch.setattr(voice_loop, "detect_wakeword", lambda *_, **__: True)
        listener._callback(np.zeros((1600, 1), dtype=np.float32), 1600, None, None)
        await asyncio.sleep(0)
        assert listener._event.is_set()

    asyncio.run(_exercise())
