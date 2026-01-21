from rex.config import load_config


def test_openai_provider_allows_none_llm_model():
    json_config = {
        "models": {
            "llm_provider": "openai",
            "llm_model": None,
        },
        "openai": {
            "model": "hermes-3-llama-3.1-8b",
            "base_url": "http://127.0.0.1:1234/v1",
        },
    }

    config = load_config(json_config=json_config, reload=True)

    assert config.llm_provider == "openai"
    assert config.llm_model is None
    assert config.openai_model == "hermes-3-llama-3.1-8b"
