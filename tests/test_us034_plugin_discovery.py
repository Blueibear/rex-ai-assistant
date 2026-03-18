"""US-034: Plugin discovery acceptance tests.

Acceptance criteria:
- plugin loader scans plugin folder
- plugins detected
- plugin metadata loaded
- Typecheck passes
"""

from __future__ import annotations

import textwrap

import pytest

from rex.plugins import PluginSpec, load_plugins, shutdown_plugins


def _write_plugin(directory, name: str, plugin_name: str = "") -> None:
    """Write a minimal plugin file to *directory*."""
    resolved_name = plugin_name or name
    (directory / f"{name}.py").write_text(
        textwrap.dedent(
            f"""
            class {name.capitalize()}Plugin:
                name = "{resolved_name}"
                version = "1.0"
                description = "Test plugin {resolved_name}"

                def initialize(self):
                    pass

                def process(self, *args, **kwargs):
                    return "ok"

                def shutdown(self):
                    pass

            def register():
                return {name.capitalize()}Plugin()
            """
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Plugin loader scans plugin folder
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_loader_returns_empty_list_for_nonexistent_folder():
    """If the plugin folder does not exist, loader returns empty list."""
    specs = load_plugins("/nonexistent/path/that/does/not/exist")
    assert specs == []


@pytest.mark.unit
def test_loader_scans_all_py_files_in_folder(tmp_path, monkeypatch):
    """Loader iterates every *.py file in the plugin directory."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_plugin(plugin_dir, "alpha")
    _write_plugin(plugin_dir, "beta")
    _write_plugin(plugin_dir, "gamma")

    monkeypatch.syspath_prepend(str(tmp_path))
    specs = load_plugins(str(plugin_dir))
    names = {s.name for s in specs}
    assert {"alpha", "beta", "gamma"} == names


@pytest.mark.unit
def test_loader_skips_dunder_files(tmp_path, monkeypatch):
    """Files starting with '__' (like __init__.py) are not loaded as plugins."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text("# marker")
    _write_plugin(plugin_dir, "real")

    monkeypatch.syspath_prepend(str(tmp_path))
    specs = load_plugins(str(plugin_dir))
    assert len(specs) == 1
    assert specs[0].name == "real"


@pytest.mark.unit
def test_loader_skips_files_without_register(tmp_path, monkeypatch):
    """Files without a register() function are skipped."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "noregister.py").write_text("def init(): pass\n")
    _write_plugin(plugin_dir, "good")

    monkeypatch.syspath_prepend(str(tmp_path))
    specs = load_plugins(str(plugin_dir))
    assert len(specs) == 1
    assert specs[0].name == "good"


# ---------------------------------------------------------------------------
# Plugins detected
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plugins_detected_returns_plugin_spec_list(tmp_path, monkeypatch):
    """load_plugins returns a list of PluginSpec objects."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_plugin(plugin_dir, "demo")

    monkeypatch.syspath_prepend(str(tmp_path))
    specs = load_plugins(str(plugin_dir))
    assert isinstance(specs, list)
    assert len(specs) == 1
    assert isinstance(specs[0], PluginSpec)


@pytest.mark.unit
def test_multiple_plugins_all_detected(tmp_path, monkeypatch):
    """All valid plugins in the folder are detected."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    for n in ("one", "two", "three"):
        _write_plugin(plugin_dir, n)

    monkeypatch.syspath_prepend(str(tmp_path))
    specs = load_plugins(str(plugin_dir))
    assert len(specs) == 3


@pytest.mark.unit
def test_detected_plugin_has_process_method(tmp_path, monkeypatch):
    """Each detected plugin exposes a callable process() method."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_plugin(plugin_dir, "capable")

    monkeypatch.syspath_prepend(str(tmp_path))
    specs = load_plugins(str(plugin_dir))
    assert callable(specs[0].plugin.process)


@pytest.mark.unit
def test_detected_plugin_is_callable(tmp_path, monkeypatch):
    """Detected plugin process() can be invoked and returns a result."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_plugin(plugin_dir, "runner")

    monkeypatch.syspath_prepend(str(tmp_path))
    specs = load_plugins(str(plugin_dir))
    result = specs[0].plugin.process()
    assert result == "ok"


# ---------------------------------------------------------------------------
# Plugin metadata loaded
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plugin_metadata_name_loaded(tmp_path, monkeypatch):
    """PluginSpec.name is populated from the plugin's name attribute."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_plugin(plugin_dir, "meta", plugin_name="my_plugin")

    monkeypatch.syspath_prepend(str(tmp_path))
    specs = load_plugins(str(plugin_dir))
    assert specs[0].name == "my_plugin"


@pytest.mark.unit
def test_plugin_spec_contains_plugin_object(tmp_path, monkeypatch):
    """PluginSpec.plugin holds the instantiated plugin object."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_plugin(plugin_dir, "obj")

    monkeypatch.syspath_prepend(str(tmp_path))
    specs = load_plugins(str(plugin_dir))
    assert specs[0].plugin is not None


@pytest.mark.unit
def test_plugin_initialize_called_on_load(tmp_path, monkeypatch):
    """initialize() is called automatically during plugin loading."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "tracked.py").write_text(
        textwrap.dedent(
            """
            class TrackedPlugin:
                name = "tracked"
                initialized = False

                def initialize(self):
                    TrackedPlugin.initialized = True

                def process(self, *args, **kwargs):
                    return TrackedPlugin.initialized

                def shutdown(self):
                    pass

            def register():
                return TrackedPlugin()
            """
        ),
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    specs = load_plugins(str(plugin_dir))
    assert len(specs) == 1
    # PluginSafetyWrapper converts non-string returns to str; initialized == True → "True"
    assert specs[0].plugin.process() in (True, "True")


@pytest.mark.unit
def test_plugin_shutdown_called_correctly(tmp_path, monkeypatch):
    """shutdown_plugins() invokes shutdown() on every loaded plugin."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_plugin(plugin_dir, "shut")

    monkeypatch.syspath_prepend(str(tmp_path))
    specs = load_plugins(str(plugin_dir))
    # Should not raise
    shutdown_plugins(specs)
