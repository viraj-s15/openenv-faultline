from training.env_adapter.client import (
    EnvUnavailableError,
    WarGamesTrainingClient,
    _RETRY_DELAYS_S,
)
from training.env_adapter.observation_formatter import build_red_prompt
from training.env_adapter.task_selector import select_curriculum_tasks
from wargames_env.models import SystemMetrics, WarGamesObservation

import httpx
import pytest


def test_build_red_prompt_includes_metrics_task_and_output():
    obs = WarGamesObservation(
        command_output="ready",
        metrics=SystemMetrics(
            gateway_success_rate=0.9,
            gateway_p99_latency_ms=120.0,
            queue_depth=3,
            worker_restart_count=1,
            consumer_stall_count=0,
        ),
        process_status={"gateway": "running"},
        done=False,
        reward=0.1,
    )

    prompt = build_red_prompt(
        observation=obs,
        task_name="phase-2-blue-l2",
        step_num=2,
        attempt_history=[{"step": 1, "command": "date", "output": "ok", "error": None}],
    )

    assert "phase-2-blue-l2" in prompt
    assert "Gateway success rate" in prompt
    assert "LATEST COMMAND OUTPUT" in prompt


def test_task_selector_uses_stage_schedule():
    schedule = [
        {"until_step": 10, "tasks": ["phase-2-blue-l0"]},
        {"until_step": 20, "tasks": ["phase-2-blue-l1"]},
    ]

    assert select_curriculum_tasks(schedule, trainer_step=5) == ["phase-2-blue-l0"]
    assert select_curriculum_tasks(schedule, trainer_step=15) == ["phase-2-blue-l1"]


def test_training_client_exposes_base_url():
    client = WarGamesTrainingClient("http://localhost:8000")

    assert client.base_url == "http://localhost:8000"
    client._client.close()


_OBS_PAYLOAD = {
    "command_output": "ok",
    "metrics": {
        "gateway_success_rate": 1.0,
        "gateway_p99_latency_ms": 10.0,
        "queue_depth": 0,
        "worker_restart_count": 0,
        "consumer_stall_count": 0,
    },
    "process_status": {},
    "done": False,
    "reward": 0.0,
}
_STEP_PAYLOAD = {"observation": _OBS_PAYLOAD, "reward": 0.5, "done": False, "info": {}}


def _client_with(handler, monkeypatch):
    # Skip real backoff sleeps in tests.
    monkeypatch.setattr("training.env_adapter.client.time.sleep", lambda _s: None)
    client = WarGamesTrainingClient("http://test")
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    return client


def test_step_retries_on_500_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500, json={"detail": "boom"})
        return httpx.Response(200, json=_STEP_PAYLOAD)

    client = _client_with(handler, monkeypatch)
    result = client.step("ls")
    assert result.reward == 0.5
    assert calls["n"] == 3


def test_step_raises_env_unavailable_after_retry_budget(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, json={"detail": "down"})

    client = _client_with(handler, monkeypatch)
    with pytest.raises(EnvUnavailableError):
        client.step("ls")
    assert calls["n"] == len(_RETRY_DELAYS_S) + 1


def test_step_does_not_retry_on_4xx(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(422, json={"detail": "bad payload"})

    client = _client_with(handler, monkeypatch)
    with pytest.raises(httpx.HTTPStatusError):
        client.step("ls")
    assert calls["n"] == 1
