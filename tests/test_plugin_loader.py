from __future__ import annotations

import textwrap

from rex.plugins import load_plugins, shutdown_plugins


def test_load_plugins_discovers_and_initialises(tmp_path, monkeypatch):
    plugin_file = tmp_path / "plugins" / "demo.py"
    plugin_file.parent.mkdir()
    plugin_file.write_text(
        textwrap.dedent(
            """
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
            """
        )
    )

    specs = load_plugins(str(plugin_file.parent))
    assert len(specs) == 1
    plugin = specs[0].plugin
    assert plugin.initialised
    assert plugin.process("hi") == "HI"

    shutdown_plugins(specs)
    assert plugin.shut_down
