from __future__ import annotations

import textwrap
from plugin_loader import load_plugins


def test_load_plugins(tmp_path, monkeypatch):
    plugin_dir = tmp_path / "test_plugins"
    plugin_dir.mkdir()

    # Make it a Python package
    (plugin_dir / "__init__.py").write_text("# package marker\n", encoding="utf-8")

    # Add one valid plugin
    (plugin_dir / "example.py").write_text(
        textwrap.dedent(
            """
            def register():
                return {"capability": "ok"}
            """
        ),
        encoding="utf-8",
    )

    # Add a second plugin (optional for scale-up testing)
    (plugin_dir / "second.py").write_text(
        textwrap.dedent(
            """
            def register():
                return {"feature": "active"}
            """
        ),
        encoding="utf-8",
    )

    # Add a broken plugin to test fallback (no register function)
    (plugin_dir / "broken.py").write_text(
        textwrap.dedent(
            """
            def init():
                return {"invalid": True}
            """
        ),
        encoding="utf-8",
    )

    # Ensure Python can import from our fake plugin path
    monkeypatch.syspath_prepend(str(tmp_path))

    # Run plugin loader
    capabilities = load_plugins(str(plugin_dir))

    # Valid plugins should be loaded
    assert isinstance(capabilities, dict), "Plugin loader should return a dict"
    assert f"{plugin_dir.name}.example" in capabilities, "example plugin should be loaded"
    assert capabilities[f"{plugin_dir.name}.example"]["capability"] == "ok", "example plugin value mismatch"

    assert f"{plugin_dir.name}.second" in capabilities, "second plugin should be loaded"
    assert capabilities[f"{plugin_dir.name}.second"]["feature"] == "active", "second plugin value mismatch"

    # Broken plugin should be skipped (or raise warning internally)
    assert f"{plugin_dir.name}.broken" not in capabilities, "broken plugin should not be loaded"

