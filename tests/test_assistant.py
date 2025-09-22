from rex.assistant import VoiceAssistant


class DummyWakeListener:
    def listen(self) -> bool:  # pragma: no cover - placeholder
        return True


class DummyLLM:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, config=None):
        self.prompts.append(prompt)
        return "assistant reply"


def test_handle_text_uses_plugin(monkeypatch):
    monkeypatch.setattr("rex.assistant.load_all_profiles", lambda: {"user": {}})
    monkeypatch.setattr("rex.assistant.load_users_map", lambda: {"user@example.com": "user"})
    monkeypatch.setattr("rex.assistant.resolve_user_key", lambda identifier, users_map, profiles=None: "user")
    history = []
    monkeypatch.setattr("rex.assistant.append_history_entry", lambda user, entry, **kwargs: history.append((user, entry)))
    monkeypatch.setattr("rex.assistant.load_recent_history", lambda user, limit=5: [])

    class DummyPlugin:
        name = "dummy"

        def initialise(self):
            return None

        def process(self, context):
            return "plugin context"

        def shutdown(self):
            return None

    llm = DummyLLM()
    assistant = VoiceAssistant(
        wake_listener=DummyWakeListener(),
        llm_client=llm,
        plugins={"dummy": DummyPlugin()},
    )

    reply = assistant.handle_text("hello")
    assert reply == "assistant reply"
    assert any(entry[1]["role"] == "user" for entry in history)
    assert any(entry[1]["role"] == "assistant" for entry in history)
    assert llm.prompts


def test_toggle_search_plugin(monkeypatch):
    monkeypatch.setattr("rex.assistant.load_all_profiles", lambda: {})
    monkeypatch.setattr("rex.assistant.load_users_map", lambda: {})
    monkeypatch.setattr("rex.assistant.resolve_user_key", lambda identifier, users_map, profiles=None: "user")
    monkeypatch.setattr("rex.assistant.extract_voice_reference", lambda profile: None)
    monkeypatch.setattr("rex.assistant.load_recent_history", lambda user, limit=5: [])
    monkeypatch.setattr("rex.assistant.append_history_entry", lambda *args, **kwargs: None)

    assistant = VoiceAssistant(
        wake_listener=DummyWakeListener(),
        llm_client=DummyLLM(),
        plugins={},
    )
    assistant.toggle_search_plugin(False)
    assert not assistant.state.enable_search
