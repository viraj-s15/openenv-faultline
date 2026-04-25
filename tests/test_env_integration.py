from pathlib import Path

from wargames_env.models import SystemMetrics, WarGamesAction
from wargames_env.client import WarGamesClient
from wargames_env.server.blue_defender import BlueAction, BlueDefenseLevel, BlueMode
from wargames_env.server.env import WarGamesEnv
from wargames_env.server.tasks import get_task_config


class FakeProcessManager:
    def __init__(self, status=None):
        self.status = status or {
            "gateway": "running pid=1",
            "auth": "running pid=2",
            "worker": "running pid=3",
            "job_generator": "running pid=4",
        }
        self.restarted = False

    def restart_all(self):
        self.restarted = True

    def wait_healthy(self, timeout_s):
        return True

    def get_status(self):
        return dict(self.status)


class StaticPoller:
    def __init__(self, metrics=None):
        self.metrics = metrics or SystemMetrics(
            gateway_success_rate=1.0,
            gateway_p99_latency_ms=0.0,
            queue_depth=0,
            worker_restart_count=0,
            consumer_stall_count=0,
        )

    def poll_once(self):
        pass

    def get_current_metrics(self):
        return self.metrics

    def stop(self):
        pass


class NoopBlueDefender:
    mode = BlueMode.SCRIPTED
    level = BlueDefenseLevel.LEVEL_0

    def tick(self, **kwargs):
        return []


def make_env(tmp_path: Path, process_manager=None, poller=None) -> WarGamesEnv:
    env = WarGamesEnv(project_root=tmp_path, mesh_root=tmp_path / "mesh")
    env._process_manager = process_manager or FakeProcessManager()
    env._metrics_poller = poller or StaticPoller()
    env._redis_flush = lambda: None
    return env


def test_task_config_matches_manifest_budget():
    config = get_task_config("phase-2-blue-l4")

    assert config.name == "phase-2-blue-l4"
    assert config.max_steps == 10


def test_reset_applies_task_config_and_exposes_task_state(tmp_path):
    env = make_env(tmp_path)

    env.reset(task_name="phase-2-blue-l4")
    state = env.state()

    assert env.max_steps == 10
    assert state["task_name"] == "phase-2-blue-l4"
    assert state["max_steps"] == 10
    assert state["blue_mode"] == "scripted"
    assert state["blue_level"] == 4


def test_step_reports_max_steps_termination_reason(tmp_path, monkeypatch):
    env = make_env(tmp_path)
    env._blue_defender = NoopBlueDefender()
    env.max_steps = 1
    monkeypatch.setattr(
        env,
        "_run_red_command",
        lambda command, timeout_s: type(
            "CommandResult",
            (),
            {
                "command": command,
                "output": "ok",
                "exit_code": 0,
                "timed_out": False,
                "duration_ms": 1,
            },
        )(),
    )

    result = env.step(WarGamesAction(command="date"))

    assert result.done is True
    assert result.info["termination_reason"] == "max_steps"


def test_step_reports_mesh_down_termination_reason(tmp_path, monkeypatch):
    process_manager = FakeProcessManager(
        status={
            "gateway": "stopped",
            "auth": "stopped",
            "worker": "stopped",
            "job_generator": "running pid=4",
        }
    )
    env = make_env(tmp_path, process_manager=process_manager)
    env._blue_defender = NoopBlueDefender()
    env.max_steps = 10
    monkeypatch.setattr(
        env,
        "_run_red_command",
        lambda command, timeout_s: type(
            "CommandResult",
            (),
            {
                "command": command,
                "output": "ok",
                "exit_code": 0,
                "timed_out": False,
                "duration_ms": 1,
            },
        )(),
    )

    result = env.step(WarGamesAction(command="date"))

    assert result.done is True
    assert result.info["termination_reason"] == "mesh_down"
    assert result.info["error"] == "critical services stopped: gateway,auth,worker"


def test_client_parses_rich_wargames_state():
    client = WarGamesClient.__new__(WarGamesClient)

    state = client._parse_state(
        {
            "episode_id": "episode-1",
            "task_name": "phase-2-blue-l4",
            "step_count": 3,
            "max_steps": 10,
            "blue_mode": "scripted",
            "blue_level": 4,
            "metrics": {
                "gateway_success_rate": 0.5,
                "gateway_p99_latency_ms": 250,
                "queue_depth": 12,
                "worker_restart_count": 1,
                "consumer_stall_count": 2,
            },
            "process_status": {"gateway": "running pid=1"},
        }
    )

    assert state.episode_id == "episode-1"
    assert state.task_name == "phase-2-blue-l4"
    assert state.step_count == 3
    assert state.max_steps == 10
    assert state.blue_mode == "scripted"
    assert state.blue_level == 4
    assert state.metrics.queue_depth == 12
    assert state.process_status == {"gateway": "running pid=1"}
