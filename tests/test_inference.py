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

    assert module._chat_token_limit_kwargs() == {"max_tokens": 4096}


def test_chat_token_limit_uses_override(monkeypatch):
    monkeypatch.setenv("CHAT_TOKEN_LIMIT_PARAM", "max_completion_tokens")
    module = importlib.reload(inference)

    assert module._chat_token_limit_kwargs() == {"max_completion_tokens": 4096}


def test_parse_tasks_reads_csv(monkeypatch):
    monkeypatch.setenv("TASKS_CSV", "phase-2-blue-l4, phase-2-blue-llm-showdown")
    module = importlib.reload(inference)

    assert module._parse_tasks() == [
        "phase-2-blue-l4",
        "phase-2-blue-llm-showdown",
    ]


def test_red_prompt_is_aggressive():
    assert "break it" in inference.SYSTEM_PROMPT
    assert "outmaneuver Blue" in inference.SYSTEM_PROMPT


def test_red_prompt_mentions_process_kill_budget():
    assert "kill, pkill, killall" in inference.SYSTEM_PROMPT
    assert "ONCE per episode" in inference.SYSTEM_PROMPT


def test_red_prompt_documents_command_chaining():
    assert "chain multiple commands" in inference.SYSTEM_PROMPT
    for op in ("`;`", "`&&`", "`||`"):
        assert op in inference.SYSTEM_PROMPT, op
    assert "stdout+stderr" in inference.SYSTEM_PROMPT


def test_red_prompt_does_not_leak_mesh_surface():
    text = inference.SYSTEM_PROMPT.lower()
    for forbidden in ("localhost:3000", "localhost:3001", "/mesh", "/process", "/verify", "registry.json"):
        assert forbidden not in text, f"system prompt leaks mesh surface: {forbidden}"



def test_kill_budget_block_reports_remaining_when_no_kill_used():
    block = inference._kill_budget_block([])
    assert "1 remaining" in block

    block = inference._kill_budget_block(
        [{"step": 1, "command": "redis-cli FLUSHALL"}]
    )
    assert "1 remaining" in block


def test_kill_budget_block_reports_spent_after_direct_kill():
    history = [
        {"step": 1, "command": "ls -la"},
        {"step": 2, "command": "kill -9 $(pgrep -f worker)"},
        {"step": 3, "command": "curl http://localhost:8080/"},
    ]
    block = inference._kill_budget_block(history)
    assert "spent on step 2" in block
    assert "will be rejected" in block


def test_kill_budget_block_detects_pkill_killall_and_xargs():
    for cmd in ("pkill -f gateway", "killall worker", "pgrep worker | xargs kill"):
        block = inference._kill_budget_block([{"step": 1, "command": cmd}])
        assert "spent on step 1" in block, cmd


def test_build_prompt_surfaces_kill_budget_above_history():
    obs = WarGamesObservation(
        metrics=SystemMetrics(
            gateway_success_rate=1.0,
            gateway_p99_latency_ms=10.0,
            queue_depth=0,
            worker_restart_count=0,
            consumer_stall_count=0,
        ),
        process_status={"gateway": "running"},
        command_output="",
    )
    prompt = inference.build_prompt(obs, 1, "phase-2-blue-l0", [])
    assert prompt.index("KILL BUDGET") < prompt.index("PREVIOUS ATTEMPTS:")






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
