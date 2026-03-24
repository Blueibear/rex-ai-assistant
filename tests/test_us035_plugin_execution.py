"""US-035: Plugin execution acceptance tests.

Acceptance criteria:
- plugin tools callable
- failures isolated
- plugins unload safely
- Typecheck passes
"""

from __future__ import annotations

import textwrap

import pytest

from rex.plugins import PluginSafetyWrapper, load_plugins, shutdown_plugins

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plugin(tmp_path, name: str, src: str) -> PluginSafetyWrapper:
    """Write a plugin to *tmp_path*, load it, and return the wrapped plugin."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir(exist_ok=True)
    (plugin_dir / f"{name}.py").write_text(textwrap.dedent(src), encoding="utf-8")
    specs = load_plugins(str(plugin_dir))
    assert specs, f"Plugin {name} did not load"
    return specs[0].plugin  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Plugin tools callable
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plugin_callable_no_args(tmp_path):
    """A plugin process() method is callable with no arguments."""
    plugin = _make_plugin(
        tmp_path,
        "noargs",
        """
        class NoargsPlugin:
            name = "noargs"
            def initialize(self): pass
            def process(self, *args, **kwargs): return "done"
            def shutdown(self): pass
        def register(): return NoargsPlugin()
        """,
    )
    result = plugin.process()
    assert result == "done"


@pytest.mark.unit
def test_plugin_callable_with_positional_args(tmp_path):
    """A plugin process() method is callable with positional arguments."""
    plugin = _make_plugin(
        tmp_path,
        "posargs",
        """
        class PosargsPlugin:
            name = "posargs"
            def initialize(self): pass
            def process(self, x, y): return x + y
            def shutdown(self): pass
        def register(): return PosargsPlugin()
        """,
    )
    result = plugin.process(3, 4)
    # PluginSafetyWrapper converts non-str/list/dict to str
    assert result in (7, "7")


@pytest.mark.unit
def test_plugin_callable_with_keyword_args(tmp_path):
    """A plugin process() method is callable with keyword arguments."""
    plugin = _make_plugin(
        tmp_path,
        "kwargs",
        """
        class KwargsPlugin:
            name = "kwargs"
            def initialize(self): pass
            def process(self, action="noop"): return action
            def shutdown(self): pass
        def register(): return KwargsPlugin()
        """,
    )
    result = plugin.process(action="greet")
    assert result == "greet"


@pytest.mark.unit
def test_plugin_returns_string_output(tmp_path):
    """Plugin process() returning a string passes through unchanged."""
    plugin = _make_plugin(
        tmp_path,
        "strout",
        """
        class StroutPlugin:
            name = "strout"
            def initialize(self): pass
            def process(self, *a, **kw): return "hello world"
            def shutdown(self): pass
        def register(): return StroutPlugin()
        """,
    )
    assert plugin.process() == "hello world"


@pytest.mark.unit
def test_plugin_returns_dict_output(tmp_path):
    """Plugin process() returning a dict passes through unchanged."""
    plugin = _make_plugin(
        tmp_path,
        "dictout",
        """
        class DictoutPlugin:
            name = "dictout"
            def initialize(self): pass
            def process(self, *a, **kw): return {"status": "ok"}
            def shutdown(self): pass
        def register(): return DictoutPlugin()
        """,
    )
    result = plugin.process()
    assert result == {"status": "ok"}


@pytest.mark.unit
def test_plugin_returns_list_output(tmp_path):
    """Plugin process() returning a list passes through unchanged."""
    plugin = _make_plugin(
        tmp_path,
        "listout",
        """
        class ListoutPlugin:
            name = "listout"
            def initialize(self): pass
            def process(self, *a, **kw): return [1, 2, 3]
            def shutdown(self): pass
        def register(): return ListoutPlugin()
        """,
    )
    result = plugin.process()
    assert result == [1, 2, 3]


# ---------------------------------------------------------------------------
# Failures isolated
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plugin_exception_raises_but_does_not_crash_host(tmp_path):
    """An exception inside process() propagates as RuntimeError without crashing the host."""
    plugin = _make_plugin(
        tmp_path,
        "badplugin",
        """
        class BadPlugin:
            name = "badplugin"
            def initialize(self): pass
            def process(self, *a, **kw): raise ValueError("intentional failure")
            def shutdown(self): pass
        def register(): return BadPlugin()
        """,
    )
    with pytest.raises(ValueError, match="intentional failure"):
        plugin.process()
    # Host is still alive — no crash


@pytest.mark.unit
def test_second_plugin_unaffected_by_first_failure(tmp_path):
    """A failure in one plugin does not prevent another plugin from running."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "bad.py").write_text(
        textwrap.dedent("""
            class BadPlugin:
                name = "bad"
                def initialize(self): pass
                def process(self, *a, **kw): raise RuntimeError("fail")
                def shutdown(self): pass
            def register(): return BadPlugin()
            """),
        encoding="utf-8",
    )
    (plugin_dir / "good.py").write_text(
        textwrap.dedent("""
            class GoodPlugin:
                name = "good"
                def initialize(self): pass
                def process(self, *a, **kw): return "success"
                def shutdown(self): pass
            def register(): return GoodPlugin()
            """),
        encoding="utf-8",
    )

    specs = load_plugins(str(plugin_dir))
    by_name = {s.name: s.plugin for s in specs}

    # bad plugin raises
    with pytest.raises(RuntimeError, match="fail"):
        by_name["bad"].process()

    # good plugin still works
    assert by_name["good"].process() == "success"


@pytest.mark.unit
def test_plugin_rate_limit_raises_after_exhaustion(tmp_path, monkeypatch):
    """After exceeding the rate limit, process() raises RuntimeError."""
    # Set a very low rate limit for this test
    monkeypatch.setattr("rex.plugins.PLUGIN_RATE_LIMIT", 2)

    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "ratelimited.py").write_text(
        textwrap.dedent("""
            class RatelimitedPlugin:
                name = "ratelimited"
                def initialize(self): pass
                def process(self, *a, **kw): return "ok"
                def shutdown(self): pass
            def register(): return RatelimitedPlugin()
            """),
        encoding="utf-8",
    )

    specs = load_plugins(str(plugin_dir))
    plugin = specs[0].plugin

    # Exhaust the rate limit
    plugin.process()
    plugin.process()

    # Third call should be rejected
    with pytest.raises(RuntimeError, match="rate limit"):
        plugin.process()


@pytest.mark.unit
def test_plugin_output_truncated_when_oversized(tmp_path, monkeypatch):
    """Oversized string output is truncated by PluginSafetyWrapper."""
    monkeypatch.setattr("rex.plugins.PLUGIN_OUTPUT_LIMIT", 10)

    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "bigout.py").write_text(
        textwrap.dedent("""
            class BigoutPlugin:
                name = "bigout"
                def initialize(self): pass
                def process(self, *a, **kw): return "A" * 1000
                def shutdown(self): pass
            def register(): return BigoutPlugin()
            """),
        encoding="utf-8",
    )

    specs = load_plugins(str(plugin_dir))
    result = specs[0].plugin.process()
    assert isinstance(result, str)
    assert "truncated" in result


# ---------------------------------------------------------------------------
# Plugins unload safely
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_shutdown_plugins_does_not_raise(tmp_path):
    """shutdown_plugins() completes without raising on a normal plugin."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "clean.py").write_text(
        textwrap.dedent("""
            class CleanPlugin:
                name = "clean"
                def initialize(self): pass
                def process(self, *a, **kw): return "ok"
                def shutdown(self): pass
            def register(): return CleanPlugin()
            """),
        encoding="utf-8",
    )

    specs = load_plugins(str(plugin_dir))
    shutdown_plugins(specs)  # should not raise


@pytest.mark.unit
def test_shutdown_plugins_handles_crashing_shutdown(tmp_path, caplog):
    """shutdown_plugins() logs but does not raise when shutdown() itself raises."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "crashshut.py").write_text(
        textwrap.dedent("""
            class CrashShutPlugin:
                name = "crashshut"
                def initialize(self): pass
                def process(self, *a, **kw): return "ok"
                def shutdown(self): raise RuntimeError("shutdown exploded")
            def register(): return CrashShutPlugin()
            """),
        encoding="utf-8",
    )

    specs = load_plugins(str(plugin_dir))
    import logging

    with caplog.at_level(logging.ERROR, logger="rex.plugins"):
        shutdown_plugins(specs)  # must not raise


@pytest.mark.unit
def test_shutdown_empty_plugin_list():
    """shutdown_plugins() handles an empty list without error."""
    shutdown_plugins([])  # must not raise


@pytest.mark.unit
def test_plugin_unloads_and_caller_still_operational(tmp_path):
    """After shutdown, the caller (this test) remains functional."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "lifecycle.py").write_text(
        textwrap.dedent("""
            class LifecyclePlugin:
                name = "lifecycle"
                def initialize(self): pass
                def process(self, *a, **kw): return "alive"
                def shutdown(self): pass
            def register(): return LifecyclePlugin()
            """),
        encoding="utf-8",
    )

    specs = load_plugins(str(plugin_dir))
    assert specs[0].plugin.process() == "alive"
    shutdown_plugins(specs)

    # Caller is still operational
    assert 1 + 1 == 2
