"""Smoke tests for rex.calendar_backends.base (CalendarBackend ABC)."""

from __future__ import annotations


def test_import():
    """Module imports without error."""
    from rex.calendar_backends import base

    assert base is not None


def test_calendar_backend_is_abstract():
    """CalendarBackend cannot be instantiated directly."""
    import pytest

    from rex.calendar_backends.base import CalendarBackend

    with pytest.raises(TypeError):
        CalendarBackend()  # type: ignore[abstract]


def _make_concrete_backend(connect_return=True, events=None):
    """Return a minimal concrete CalendarBackend subclass."""
    from rex.calendar_backends.base import CalendarBackend

    class _Concrete(CalendarBackend):
        def __init__(self):
            self._connected = False

        def connect(self) -> bool:
            self._connected = connect_return
            return connect_return

        def fetch_events(self):
            return events or []

    return _Concrete()


def test_connect_returns_bool():
    """Concrete backend's connect() returns a bool."""
    backend = _make_concrete_backend(connect_return=True)
    result = backend.connect()
    assert result is True


def test_fetch_events_returns_list():
    """Concrete backend's fetch_events() returns a list."""
    backend = _make_concrete_backend(events=[])
    events = backend.fetch_events()
    assert isinstance(events, list)


def test_disconnect_default_no_error():
    """Default disconnect() is a no-op and does not raise."""
    backend = _make_concrete_backend()
    backend.disconnect()  # should not raise


def test_is_connected_default_false():
    """Default is_connected property returns False."""
    backend = _make_concrete_backend()
    assert backend.is_connected is False


def test_backend_name_returns_class_name():
    """backend_name returns the concrete class name."""
    backend = _make_concrete_backend()
    assert "_Concrete" in backend.backend_name


def test_test_connection_success():
    """test_connection returns (True, None) when connect() succeeds."""
    backend = _make_concrete_backend(connect_return=True)
    ok, msg = backend.test_connection()
    assert ok is True
    assert msg is None


def test_test_connection_failure_false():
    """test_connection returns (False, message) when connect() returns False."""
    backend = _make_concrete_backend(connect_return=False)
    ok, msg = backend.test_connection()
    assert ok is False
    assert msg is not None


def test_test_connection_exception():
    """test_connection returns (False, error_str) when connect() raises."""
    from rex.calendar_backends.base import CalendarBackend

    class _Failing(CalendarBackend):
        def connect(self):
            raise RuntimeError("no connection")

        def fetch_events(self):
            return []

    backend = _Failing()
    ok, msg = backend.test_connection()
    assert ok is False
    assert "no connection" in msg
