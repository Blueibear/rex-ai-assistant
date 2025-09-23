from __future__ import annotations

import asyncio

import rex.assistant as assistant_module


class DummyStrategy:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, config=None):
        self.calls.append(prompt)
        return "hello"


class DummyPlugin:
    name = "dummy"

    def __init__(self):
        self.initialised = True

    def initialize(self):
        self.initialised = True

    def process(self, transcript):
        return "plugin info"

    def shutdown(self):
        pass


async def _run_assistant(assistant):
    return await assistant.generate_reply("hi")


def test_assistant_generates_reply(monkeypatch, tmp_path):
    dummy_strategy = DummyStrategy()

    class DummyLanguageModel:
        def __init__(self, *args, **kwargs):
            self.strategy = dummy_strategy

        def generate(self, prompt, config=None):
            return dummy_strategy.generate(prompt, config)

    monkeypatch.setattr(assistant_module, "LanguageModel", DummyLanguageModel)

    plugin_spec = assistant_module.PluginSpec(name="dummy", plugin=DummyPlugin())
    assistant = assistant_module.Assistant(plugins=[plugin_spec], transcripts_dir=tmp_path)

    reply = asyncio.run(_run_assistant(assistant))

    assert "hello" in reply
    assert "plugin info" in reply
    assert any("user:" in call for call in dummy_strategy.calls)
