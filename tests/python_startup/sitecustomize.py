"""Pytest subprocess startup compatibility."""

from __future__ import annotations

import sys
import time

_SLEEP = time.sleep


def _install_asyncio_fallback() -> None:
    if sys.platform != "win32":
        return

    try:
        import _overlapped  # noqa: F401
    except Exception:
        pass
    else:
        return

    if "asyncio" not in sys.modules:
        real_platform = sys.platform
        try:
            sys.platform = "linux"
            import asyncio  # noqa: F401
        finally:
            sys.platform = real_platform

    import asyncio

    class _DummySelector:
        def __init__(self) -> None:
            self._map = {}

        def register(self, fileobj, events, data=None):
            self._map[fileobj] = (fileobj, events, data)
            return self._map[fileobj]

        def unregister(self, fileobj):
            return self._map.pop(fileobj, None)

        def modify(self, fileobj, events, data=None):
            self._map[fileobj] = (fileobj, events, data)
            return self._map[fileobj]

        def select(self, timeout=None):
            if timeout is None:
                timeout = 0.001
            if timeout > 0:
                _SLEEP(min(timeout, 0.001))
            return []

        def get_map(self):
            return self._map

        def close(self) -> None:
            self._map.clear()

    class _Loop(asyncio.SelectorEventLoop):
        def __init__(self) -> None:
            super().__init__(selector=_DummySelector())

        def _make_self_pipe(self) -> None:
            self._ssock = None
            self._csock = None
            self._internal_fds = 0

        def _close_self_pipe(self) -> None:
            pass

        def _write_to_self(self) -> None:
            pass

    class _Policy(asyncio.DefaultEventLoopPolicy):
        def new_event_loop(self):
            return _Loop()

    asyncio.set_event_loop_policy(_Policy())


_install_asyncio_fallback()


def _install_ssl_fallback() -> None:
    try:
        import ssl
    except Exception:
        return

    original = ssl.create_default_context

    def create_default_context(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except NameError as exc:
            if "enum_certificates" not in str(exc):
                raise
            return ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    ssl.create_default_context = create_default_context


_install_ssl_fallback()
