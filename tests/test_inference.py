import importlib

import inference


def test_extract_action_payload_reads_embedded_json():
    command, reasoning = inference.extract_action_payload(
        'Here is the action:\n{"command":"redis-cli KEYS *","reasoning":"recon"}'
    )

    assert command == "redis-cli KEYS *"
    assert reasoning == "recon"


def test_chat_token_limit_uses_max_tokens_for_hf_router(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.delenv("CHAT_TOKEN_LIMIT_PARAM", raising=False)
    module = importlib.reload(inference)

    assert module._chat_token_limit_kwargs() == {"max_tokens": 2048}


def test_chat_token_limit_uses_override(monkeypatch):
    monkeypatch.setenv("CHAT_TOKEN_LIMIT_PARAM", "max_completion_tokens")
    module = importlib.reload(inference)

    assert module._chat_token_limit_kwargs() == {"max_completion_tokens": 2048}


def test_parse_tasks_reads_csv(monkeypatch):
    monkeypatch.setenv("TASKS_CSV", "phase-2-blue-l4, phase-2-blue-llm-showdown")
    module = importlib.reload(inference)

    assert module._parse_tasks() == [
        "phase-2-blue-l4",
        "phase-2-blue-llm-showdown",
    ]
