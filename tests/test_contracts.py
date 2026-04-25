import subprocess

import pytest
from fastapi.testclient import TestClient

from wargames_env.client import WarGamesClient
from wargames_env.models import (
    StepResult,
    SystemMetrics,
    WarGamesAction,
    WarGamesObservation,
)
from wargames_env.server.app import app
from wargames_env.server.env import WarGamesEnv
from wargames_env.server.metrics_poller import MetricsPoller
from wargames_env.server.process_manager import ProcessManager


def test_models_expose_bash_action_and_metrics_observation():
    action = WarGamesAction(command="curl localhost:3000/health")
    metrics = SystemMetrics(
        gateway_success_rate=1.0,
        gateway_p99_latency_ms=12.5,
        queue_depth=0,
        worker_restart_count=0,
        consumer_stall_count=0,
    )
    observation = WarGamesObservation(
        command_output="ready",
        metrics=metrics,
        process_status={"gateway": "running"},
        done=False,
        reward=0.0,
    )

    assert action.command == "curl localhost:3000/health"
    assert observation.metrics.gateway_success_rate == 1.0
    assert observation.process_status["gateway"] == "running"


def test_environment_state_starts_at_zero_steps():
    env = WarGamesEnv()
    state = env.state()

    assert state["step_count"] == 0
    assert state["episode_id"]


def test_fastapi_app_uses_wargames_title():
    assert "WarGames" in app.title


def test_root_server_module_exports_package_app():
    from server.app import app as root_app

    assert root_app is app


def test_close_does_not_stop_shared_mesh_processes():
    class FakeProcessManager:
        def __init__(self):
            self.stop_calls = 0

        def stop_all(self):
            self.stop_calls += 1

    env = WarGamesEnv()
    fake_process_manager = FakeProcessManager()
    env._process_manager = fake_process_manager

    env.close()

    assert fake_process_manager.stop_calls == 0


def test_environment_exposes_round_one_style_methods():
    env = WarGamesEnv()

    assert callable(env.reset)
    assert callable(env.step)
    assert callable(env.state)


def test_action_rejects_empty_command():
    with pytest.raises(ValueError):
        WarGamesAction(command="   ")


def test_client_serializes_action_and_parses_step_result():
    client = WarGamesClient.__new__(WarGamesClient)
    payload = {
        "observation": {
            "command_output": "ok",
            "metrics": {
                "gateway_success_rate": 0.75,
                "gateway_p99_latency_ms": 42,
                "queue_depth": 3,
                "worker_restart_count": 1,
                "consumer_stall_count": 2,
            },
            "process_status": {"gateway": "running pid=123"},
        },
        "reward": 0.5,
        "done": False,
    }

    assert client._step_payload(WarGamesAction(command="date")) == {"command": "date"}
    result = client._parse_result(payload)

    assert isinstance(result, StepResult)
    assert result.observation.command_output == "ok"
    assert result.observation.metrics.queue_depth == 3
    assert result.reward == 0.5


def test_client_parses_state_payload():
    client = WarGamesClient.__new__(WarGamesClient)

    state = client._parse_state({"episode_id": "episode-1", "step_count": 7})

    assert state.episode_id == "episode-1"
    assert state.step_count == 7


def test_metrics_poller_updates_snapshot_from_sources(monkeypatch):
    poller = MetricsPoller()
    monkeypatch.setattr(
        poller,
        "_poll_gateway",
        lambda: {"gateway_success_rate": 0.9, "gateway_p99_latency_ms": 25.0},
    )
    monkeypatch.setattr(poller, "_poll_queue_depth", lambda: 11)
    monkeypatch.setattr(
        poller,
        "_read_counter",
        lambda path: 4 if path == "/tmp/worker_restart_count" else 5,
    )

    poller.poll_once()
    metrics = poller.get_current_metrics()

    assert metrics.gateway_success_rate == 0.9
    assert metrics.gateway_p99_latency_ms == 25.0
    assert metrics.queue_depth == 11
    assert metrics.worker_restart_count == 4
    assert metrics.consumer_stall_count == 5


def test_metrics_poller_queue_depth_falls_back_on_redis_error(monkeypatch):
    poller = MetricsPoller()
    poller._latest["queue_depth"] = 8

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args, returncode=1, stdout="", stderr="err"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert poller._poll_queue_depth() == 8


def test_process_manager_defaults_to_repo_mesh(tmp_path):
    manager = ProcessManager(project_root=tmp_path)

    assert manager.project_root == tmp_path
    assert manager.mesh_root == tmp_path / "mesh"
    assert manager._service_scripts["gateway"] == tmp_path / "mesh/gateway/index.ts"


def test_process_manager_status_uses_pid_reader(tmp_path, monkeypatch):
    manager = ProcessManager(project_root=tmp_path)
    pids = {"gateway": 1234, "worker": 5678}
    monkeypatch.setattr(manager, "_read_pid", lambda service: pids.get(service))

    status = manager.get_status()

    assert status["gateway"] == "running pid=1234"
    assert status["auth"] == "stopped"
    assert status["worker"] == "running pid=5678"
    assert status["job_generator"] == "stopped"


def test_environment_reset_uses_process_manager_and_returns_metrics(
    tmp_path, monkeypatch
):
    class FakeProcessManager:
        def __init__(self):
            self.restarted = False

        def restart_all(self):
            self.restarted = True

        def wait_healthy(self, timeout_s):
            return timeout_s == 30

        def get_status(self):
            return {"gateway": "running"}

    class FakePoller:
        def __init__(self):
            self.polled = False

        def poll_once(self):
            self.polled = True

        def get_current_metrics(self):
            return SystemMetrics(
                gateway_success_rate=1.0,
                gateway_p99_latency_ms=0.0,
                queue_depth=0,
                worker_restart_count=0,
                consumer_stall_count=0,
            )

        def stop(self):
            pass

    env = WarGamesEnv(project_root=tmp_path, mesh_root=tmp_path / "mesh")
    process_manager = FakeProcessManager()
    poller = FakePoller()
    env._process_manager = process_manager
    env._metrics_poller = poller
    monkeypatch.setattr(env, "_redis_flush", lambda: None)

    observation = env.reset()

    assert process_manager.restarted is True
    assert poller.polled is True
    assert observation.command_output == "WarGames mesh ready."
    assert observation.metrics.gateway_success_rate == 1.0


def test_environment_step_returns_output_metrics_and_exit_code(tmp_path, monkeypatch):
    class FakeProcessManager:
        def get_status(self):
            return {"gateway": "running"}

    class FakePoller:
        def poll_once(self):
            pass

        def get_current_metrics(self):
            return SystemMetrics(
                gateway_success_rate=1.0,
                gateway_p99_latency_ms=0.0,
                queue_depth=2,
                worker_restart_count=0,
                consumer_stall_count=0,
            )

        def stop(self):
            pass

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="hello\n", stderr=""
        )

    env = WarGamesEnv(project_root=tmp_path, mesh_root=tmp_path / "mesh")
    env._process_manager = FakeProcessManager()
    env._metrics_poller = FakePoller()
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = env.step(WarGamesAction(command="echo hello"))

    assert result.observation.command_output == "hello"
    assert result.observation.metrics.queue_depth == 2
    assert result.reward == 0.0
    assert result.info["exit_code"] == 0


def test_fastapi_routes_delegate_to_env():
    class FakeEnv:
        def reset(self, task_name=None):
            return WarGamesObservation(
                command_output=f"reset:{task_name}",
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

        def step(self, action):
            return StepResult(
                observation=self.reset(),
                reward=0.0,
                done=False,
                info={"command": action.command},
            )

        def state(self):
            return {"episode_id": "episode-1", "step_count": 1}

    app.state.env = FakeEnv()
    client = TestClient(app)

    reset_response = client.post("/reset?task_name=demo")
    step_response = client.post("/step", json={"command": "date"})
    state_response = client.get("/state")

    assert reset_response.status_code == 200
    assert reset_response.json()["command_output"] == "reset:demo"
    assert step_response.status_code == 200
    assert step_response.json()["info"]["command"] == "date"
    assert state_response.json()["step_count"] == 1
