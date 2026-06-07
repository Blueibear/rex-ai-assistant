from __future__ import annotations

import textwrap

import pytest

from rex.plugins import load_plugins as load_rex_plugins
from rex.plugins import shutdown_plugins


@pytest.mark.unit
def test_class_based_plugin_loads_and_runs(tmp_path, monkeypatch):
    plugin_file = tmp_path / "plugins" / "demo.py"
    plugin_file.parent.mkdir(parents=True)
    plugin_file.write_text(textwrap.dedent("""
            from rex.plugins import Plugin

            class DemoPlugin:
                name = "demo"

                def __init__(self):
                    self.initialised = False
                    self.shut_down = False

                def initialize(self):
                    self.initialised = True

                def process(self, value):
                    return value.upper()

                def shutdown(self):
                    self.shut_down = True

            def register() -> Plugin:
                return DemoPlugin()
            """))

    monkeypatch.syspath_prepend(str(tmp_path))
    specs = load_rex_plugins(str(plugin_file.parent))
    assert len(specs) == 1

    plugin = specs[0].plugin
    assert plugin.initialised
    assert plugin.process("hi") == "HI"

    shutdown_plugins(specs)
    assert plugin.shut_down


@pytest.mark.unit
def test_retired_dict_plugin_loader_not_imported():
    """The legacy dict-based rex.plugin_loader module has been retired."""
    import importlib.util

    assert importlib.util.find_spec("rex.plugin_loader") is None
