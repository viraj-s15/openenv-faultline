from training.env_adapter.client import WarGamesTrainingClient
from training.env_adapter.observation_formatter import build_red_prompt
from training.env_adapter.task_selector import select_curriculum_tasks
from wargames_env.models import SystemMetrics, WarGamesObservation


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
