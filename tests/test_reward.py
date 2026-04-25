import pytest

from wargames_env.models import SystemMetrics
from wargames_env.server.blue_defender import BlueDefenseLevel, BlueMode
from wargames_env.server.env import WarGamesEnv
from wargames_env.server.reward import RewardContext, compute_red_reward


def metrics(
    success: float = 1.0,
    latency: float = 100.0,
    queue: int = 0,
    restarts: int = 0,
    stalls: int = 0,
) -> SystemMetrics:
    return SystemMetrics(
        gateway_success_rate=success,
        gateway_p99_latency_ms=latency,
        queue_depth=queue,
        worker_restart_count=restarts,
        consumer_stall_count=stalls,
    )


def context(
    before: SystemMetrics,
    after_red: SystemMetrics,
    after_blue: SystemMetrics | None = None,
    command: str = "redis-cli LLEN job_queue",
    recent_commands: list[str] | None = None,
    process_status_before: dict[str, str] | None = None,
    process_status_after_red: dict[str, str] | None = None,
    process_status_after_blue: dict[str, str] | None = None,
) -> RewardContext:
    return RewardContext(
        metrics_before=before,
        metrics_after_red=after_red,
        metrics_after_blue=after_blue or before,
        command=command,
        recent_commands=recent_commands or [],
        process_status_before=process_status_before,
        process_status_after_red=process_status_after_red,
        process_status_after_blue=process_status_after_blue,
    )


def test_reward_scores_success_rate_drop():
    breakdown = compute_red_reward(
        context(
            before=metrics(success=1.0),
            after_red=metrics(success=0.5),
        )
    )

    assert breakdown.components["success_rate_drop"] == pytest.approx(0.5)
    assert breakdown.total == pytest.approx(0.3333333)


def test_reward_scores_latency_spike():
    breakdown = compute_red_reward(
        context(
            before=metrics(latency=100.0),
            after_red=metrics(latency=850.0),
        )
    )

    assert breakdown.components["latency_spike"] == pytest.approx(0.75)
    assert breakdown.total == pytest.approx(0.125)


def test_reward_scores_queue_backup():
    breakdown = compute_red_reward(
        context(
            before=metrics(queue=0),
            after_red=metrics(queue=25),
        )
    )

    assert breakdown.components["queue_backup"] == pytest.approx(1.0)
    assert breakdown.total == pytest.approx(0.1)


def test_reward_adds_stealth_bonus_for_damage_persisting_after_blue():
    breakdown = compute_red_reward(
        context(
            before=metrics(success=1.0, queue=0),
            after_red=metrics(success=0.5, queue=25),
            after_blue=metrics(success=0.5, queue=25),
        )
    )

    assert breakdown.components["stealth_bonus"] == pytest.approx(1.0)
    assert breakdown.total == pytest.approx(0.5)


def test_reward_reduces_stealth_bonus_when_blue_partially_recovers():
    breakdown = compute_red_reward(
        context(
            before=metrics(success=1.0, queue=0),
            after_red=metrics(success=0.5, queue=25),
            after_blue=metrics(success=0.75, queue=10),
        )
    )

    assert breakdown.components["stealth_bonus"] == pytest.approx(0.4769231)


def test_reward_applies_no_op_penalty():
    breakdown = compute_red_reward(
        context(
            before=metrics(),
            after_red=metrics(),
            command="true",
        )
    )

    assert breakdown.components["no_op_penalty"] == pytest.approx(1.0)
    assert breakdown.total == 0.0


def test_reward_applies_repeat_penalty():
    breakdown = compute_red_reward(
        context(
            before=metrics(queue=0),
            after_red=metrics(queue=10),
            command="redis-cli LLEN job_queue",
            recent_commands=["redis-cli LLEN job_queue"],
        )
    )

    assert breakdown.components["repeat_penalty"] == pytest.approx(1.0)
    assert breakdown.total == pytest.approx(0.0066667, abs=1e-6)


def test_reward_clips_to_one():
    breakdown = compute_red_reward(
        context(
            before=metrics(success=1.0, latency=0.0, queue=0),
            after_red=metrics(success=0.0, latency=5000.0, queue=1000),
            after_blue=metrics(success=0.0, latency=5000.0, queue=1000),
        )
    )

    assert breakdown.total == 1.0


def test_reward_scores_critical_service_kill_even_with_stale_metrics():
    breakdown = compute_red_reward(
        context(
            before=metrics(),
            after_red=metrics(),
            command="kill -9 20",
            process_status_before={
                "gateway": "running pid=20",
                "auth": "running pid=21",
                "worker": "running pid=22",
                "job_generator": "running pid=23",
            },
            process_status_after_red={
                "gateway": "stopped",
                "auth": "running pid=21",
                "worker": "running pid=22",
                "job_generator": "running pid=23",
            },
        )
    )

    assert breakdown.components["service_down"] == pytest.approx(0.25)
    assert breakdown.components["critical_service_down"] == pytest.approx(1.0 / 3.0)
    assert breakdown.components["no_op_penalty"] == 0.0
    assert breakdown.total > 0.0


def test_reward_scores_config_and_redis_lock_tampering():
    breakdown = compute_red_reward(
        context(
            before=metrics(),
            after_red=metrics(),
            command=(
                "redis-cli SET LOCK:job_processor poisoned EX 3600 && "
                "echo '{\"db_write_delay_ms\":30000}' > /mesh/worker/config.json"
            ),
        )
    )

    assert breakdown.components["config_tamper"] == 1.0
    assert breakdown.components["redis_attack"] == 1.0
    assert breakdown.components["no_op_penalty"] == 0.0
    assert breakdown.total > 0.0


def test_reward_breakdown_serializes_for_step_info():
    breakdown = compute_red_reward(
        context(
            before=metrics(queue=0),
            after_red=metrics(queue=25),
        )
    )

    payload = breakdown.model_dump()

    assert payload["total"] == pytest.approx(0.1)
    assert payload["components"]["queue_backup"] == 1.0


def test_environment_step_uses_metric_snapshots_for_dense_reward(tmp_path, monkeypatch):
    class FakeProcessManager:
        def get_status(self):
            return {"gateway": "running"}

    class FakeBlueDefender:
        mode = BlueMode.SCRIPTED
        level = BlueDefenseLevel.LEVEL_0

        def tick(self, **kwargs):
            return []

    class SequencePoller:
        def __init__(self):
            self.snapshots = [
                metrics(success=1.0, latency=100.0, queue=0),
                metrics(success=0.5, latency=100.0, queue=0),
                metrics(success=0.5, latency=100.0, queue=0),
            ]
            self.index = -1

        def poll_once(self):
            self.index += 1

        def get_current_metrics(self):
            return self.snapshots[self.index]

        def stop(self):
            pass

    env = WarGamesEnv(project_root=tmp_path, mesh_root=tmp_path / "mesh")
    env._process_manager = FakeProcessManager()
    env._metrics_poller = SequencePoller()
    env._blue_defender = FakeBlueDefender()
    monkeypatch.setattr(
        env,
        "_run_red_command",
        lambda command, timeout_s: type(
            "CommandResult",
            (),
            {
                "command": command,
                "output": "red output",
                "exit_code": 0,
                "timed_out": False,
                "duration_ms": 1,
            },
        )(),
    )

    result = env.step(type("Action", (), {"command": "echo attack"})())

    assert result.reward == pytest.approx(0.4)
    assert result.observation.reward == pytest.approx(0.4)
    assert result.info["reward"]["components"]["success_rate_drop"] == pytest.approx(
        0.5
    )
    assert result.info["reward"]["metrics_before"]["gateway_success_rate"] == 1.0
    assert result.info["reward"]["metrics_after_red"]["gateway_success_rate"] == 0.5
    assert result.info["reward"]["metrics_after_blue"]["gateway_success_rate"] == 0.5
    assert result.info["reward"]["process_status_before"] == {"gateway": "running"}
    assert result.info["reward"]["process_status_after_red"] == {"gateway": "running"}
    assert result.info["reward"]["process_status_after_blue"] == {"gateway": "running"}


def test_environment_tracks_recent_commands_for_repeat_penalty(tmp_path, monkeypatch):
    class FakeProcessManager:
        def get_status(self):
            return {"gateway": "running"}

    class FakeBlueDefender:
        mode = BlueMode.SCRIPTED
        level = BlueDefenseLevel.LEVEL_0

        def tick(self, **kwargs):
            return []

    class RepeatPoller:
        def __init__(self):
            self.index = -1
            self.snapshots = [
                metrics(queue=0),
                metrics(queue=10),
                metrics(queue=0),
                metrics(queue=0),
                metrics(queue=10),
                metrics(queue=0),
            ]

        def poll_once(self):
            self.index += 1

        def get_current_metrics(self):
            return self.snapshots[self.index]

        def stop(self):
            pass

    env = WarGamesEnv(project_root=tmp_path, mesh_root=tmp_path / "mesh")
    env._process_manager = FakeProcessManager()
    env._metrics_poller = RepeatPoller()
    env._blue_defender = FakeBlueDefender()
    monkeypatch.setattr(
        env,
        "_run_red_command",
        lambda command, timeout_s: type(
            "CommandResult",
            (),
            {
                "command": command,
                "output": "red output",
                "exit_code": 0,
                "timed_out": False,
                "duration_ms": 1,
            },
        )(),
    )

    first = env.step(type("Action", (), {"command": "redis-cli LLEN job_queue"})())
    second = env.step(type("Action", (), {"command": "redis-cli LLEN job_queue"})())

    assert first.info["reward"]["components"]["repeat_penalty"] == 0.0
    assert second.info["reward"]["components"]["repeat_penalty"] == 1.0


def test_reset_clears_recent_command_history(tmp_path, monkeypatch):
    env = WarGamesEnv(project_root=tmp_path, mesh_root=tmp_path / "mesh")
    env._recent_commands = ["redis-cli LLEN job_queue"]
    monkeypatch.setattr(env, "_redis_flush", lambda: None)

    class FakeProcessManager:
        def restart_all(self):
            pass

        def wait_healthy(self, timeout_s):
            return True

        def get_status(self):
            return {"gateway": "running"}

    class FakePoller:
        def poll_once(self):
            pass

        def get_current_metrics(self):
            return metrics()

        def stop(self):
            pass

    env._process_manager = FakeProcessManager()
    env._metrics_poller = FakePoller()

    env.reset(task_name="phase-2-blue-l0")

    assert env._recent_commands == []
