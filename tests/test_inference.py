import importlib
from types import SimpleNamespace

import inference
from wargames_env.models import SystemMetrics, WarGamesObservation


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


def test_red_prompt_mentions_proactive_blue_defense():
    assert "proactively harden" in inference.SYSTEM_PROMPT
    assert "monitor" in inference.SYSTEM_PROMPT
    assert "restore" in inference.SYSTEM_PROMPT
    assert "sanitize" in inference.SYSTEM_PROMPT


def test_red_prompt_mentions_process_kill_budget():
    assert "Direct process-kill commands are limited to one use per episode" in (
        inference.SYSTEM_PROMPT
    )


def test_run_episode_uses_server_state_max_steps(monkeypatch):
    class FakeLLMClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create_completion)
            )

        def create_completion(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"command":"date","reasoning":"probe"}'
                        )
                    )
                ]
            )

    class FakeEnv:
        def __init__(self):
            self.step_calls = 0
            self.actions = []

        def reset(self, task_name):
            return WarGamesObservation(
                command_output="ready",
                metrics=SystemMetrics(
                    gateway_success_rate=1.0,
                    gateway_p99_latency_ms=0.0,
                    queue_depth=0,
                    worker_restart_count=0,
                    consumer_stall_count=0,
                ),
                process_status={"gateway": "running"},
                done=False,
                reward=0.0,
            )

        def state(self):
            return SimpleNamespace(max_steps=2)

        def step(self, action):
            self.step_calls += 1
            self.actions.append(action)
            return SimpleNamespace(
                observation=self.reset("phase-2-blue-l4"),
                reward=0.0,
                done=False,
                info={},
            )

    fake_env = FakeEnv()
    monkeypatch.setattr(inference, "MAX_STEPS_CAP", 0)

    inference._run_episode(FakeLLMClient(), fake_env, "phase-2-blue-l4")

    assert fake_env.step_calls == 2
    assert fake_env.actions[0].reasoning == "probe"
