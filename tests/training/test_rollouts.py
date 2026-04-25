from types import SimpleNamespace

from training.rollouts.episode_runner import run_episode


class FakeLLM:
    def __init__(self, outputs):
        self._outputs = iter(outputs)

    def generate(self, prompt: str) -> str:
        return next(self._outputs)


class FakeTrainingClient:
    def __init__(self):
        self.step_calls = 0

    def reset(self, task_name: str):
        return SimpleNamespace(
            command_output="ready",
            metrics=SimpleNamespace(
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

    def step(self, command: str):
        self.step_calls += 1
        return SimpleNamespace(
            observation=self.reset("phase-2-blue-l0"),
            reward=0.25,
            done=self.step_calls >= 2,
            info={"error": None},
        )


def test_run_episode_returns_steps_rewards_and_transcript():
    result = run_episode(
        llm_client=FakeLLM([
            '{"command":"date"}',
            '{"command":"redis-cli LLEN job_queue"}',
        ]),
        env_client=FakeTrainingClient(),
        task_name="phase-2-blue-l0",
        max_steps=2,
    )

    assert result.task_name == "phase-2-blue-l0"
    assert len(result.steps) == 2
    assert len(result.rewards) == 2
    assert result.rewards == [0.25, 0.25]
