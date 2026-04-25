import subprocess
from pathlib import Path

from wargames_env.models import SystemMetrics
from wargames_env.server.blue_defender import (
    BlueAction,
    BlueDefender,
    BlueDefenseLevel,
    BlueMode,
    BlueSelection,
    select_blue_defender,
)
from wargames_env.server.blue_llm import BLUE_SYSTEM_PROMPT, build_blue_prompt
from wargames_env.server.config_baseline import ConfigBaseline
from wargames_env.server.env import WarGamesEnv


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
        return SystemMetrics(
            gateway_success_rate=1.0,
            gateway_p99_latency_ms=0.0,
            queue_depth=0,
            worker_restart_count=0,
            consumer_stall_count=0,
        )

    def stop(self):
        pass


def make_env(tmp_path: Path) -> WarGamesEnv:
    env = WarGamesEnv(project_root=tmp_path, mesh_root=tmp_path / "mesh")
    env._process_manager = FakeProcessManager()
    env._metrics_poller = FakePoller()
    env._redis_flush = lambda: None
    return env


def test_select_blue_defender_defaults_unknown_tasks_to_scripted_level_zero():
    selection = select_blue_defender("unknown-task")

    assert selection.mode == BlueMode.SCRIPTED
    assert selection.level == BlueDefenseLevel.LEVEL_0
    assert selection.task_name == "unknown-task"


def test_select_blue_defender_maps_scripted_curriculum_tasks():
    selection = select_blue_defender("phase-2-blue-l3")

    assert selection.mode == BlueMode.SCRIPTED
    assert selection.level == BlueDefenseLevel.LEVEL_3


def test_select_blue_defender_maps_llm_showdown_to_single_hardest_mode():
    selection = select_blue_defender("phase-2-blue-llm-showdown")

    assert selection.mode == BlueMode.LLM_SHOWDOWN
    assert selection.level == BlueDefenseLevel.LEVEL_4


def test_reset_exposes_blue_state_and_current_task(tmp_path):
    env = make_env(tmp_path)

    env.reset(task_name="phase-2-blue-l2")
    state = env.state()

    assert state["blue_mode"] == "scripted"
    assert state["blue_level"] == 2
    assert Path("/tmp/current_task").read_text(encoding="utf-8") == "phase-2-blue-l2"


def test_step_runs_blue_tick_and_returns_blue_actions(tmp_path, monkeypatch):
    class FakeBlueDefender:
        mode = BlueMode.SCRIPTED
        level = BlueDefenseLevel.LEVEL_1

        def __init__(self):
            self.received = None

        def tick(self, **kwargs):
            self.received = kwargs
            return [
                BlueAction(
                    kind="restart",
                    target="worker",
                    status="applied",
                    detail="worker restarted",
                )
            ]

    env = make_env(tmp_path)
    blue_defender = FakeBlueDefender()
    env._blue_defender = blue_defender

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

    result = env.step(
        action=type("Action", (), {"command": "date", "reasoning": "check clock"})()
    )

    assert blue_defender.received["red_command"] == "date"
    assert "red_reasoning" not in blue_defender.received
    assert blue_defender.received["process_manager"] is env._process_manager
    assert result.info["blue_actions"] == [
        {
            "kind": "restart",
            "target": "worker",
            "status": "applied",
            "detail": "worker restarted",
        }
    ]


def test_llm_showdown_runs_two_blue_ticks_per_red_step(tmp_path, monkeypatch):
    class FakeBlueDefender:
        mode = BlueMode.LLM_SHOWDOWN
        level = BlueDefenseLevel.LEVEL_4

        def __init__(self):
            self.calls = 0

        def tick(self, **kwargs):
            self.calls += 1
            return [
                BlueAction(
                    kind="llm_command",
                    target=f"repair-{self.calls}",
                    status="applied",
                    detail=f"turn {self.calls}",
                )
            ]

    env = make_env(tmp_path)
    blue_defender = FakeBlueDefender()
    env._blue_defender = blue_defender

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

    result = env.step(action=type("Action", (), {"command": "date"})())

    assert blue_defender.calls == 2
    assert result.info["blue_actions"] == [
        {
            "kind": "llm_command",
            "target": "repair-1",
            "status": "applied",
            "detail": "turn 1",
        },
        {
            "kind": "llm_command",
            "target": "repair-2",
            "status": "applied",
            "detail": "turn 2",
        },
    ]


def test_scripted_level_zero_takes_no_blue_actions(tmp_path):
    defender = BlueDefender(
        BlueSelection(
            mode=BlueMode.SCRIPTED,
            level=BlueDefenseLevel.LEVEL_0,
            task_name="phase-2-blue-l0",
        )
    )

    assert defender.tick(process_manager=FakeProcessManager(), mesh_root=tmp_path) == []


def test_scripted_level_one_starts_stopped_services(tmp_path):
    class RestartingProcessManager:
        def __init__(self):
            self.started = False

        def get_status(self):
            return {
                "gateway": "running pid=1",
                "auth": "running pid=2",
                "worker": "stopped",
                "job_generator": "stopped",
            }

        def start_all(self):
            self.started = True

    manager = RestartingProcessManager()
    defender = BlueDefender(
        BlueSelection(
            mode=BlueMode.SCRIPTED,
            level=BlueDefenseLevel.LEVEL_1,
            task_name="phase-2-blue-l1",
        )
    )

    actions = defender.tick(process_manager=manager, mesh_root=tmp_path)

    assert manager.started is True
    assert actions == [
        BlueAction(
            kind="restart",
            target="worker,job_generator",
            status="applied",
            detail="started stopped services",
        )
    ]


def test_config_baseline_restores_modified_config(tmp_path):
    mesh_root = tmp_path / "mesh"
    config_path = mesh_root / "auth" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text('{"delay_ms":200}\n', encoding="utf-8")
    baseline = ConfigBaseline.capture(mesh_root)

    config_path.write_text('{"delay_ms":1500}\n', encoding="utf-8")

    restored = baseline.restore_modified()

    assert restored == [config_path]
    assert config_path.read_text(encoding="utf-8") == '{"delay_ms":200}\n'


def test_scripted_level_two_restores_configs_and_sighups_services(tmp_path):
    class SighupProcessManager(FakeProcessManager):
        def __init__(self):
            self.signaled = []

        def get_status(self):
            return {
                "gateway": "running pid=1",
                "auth": "running pid=2",
                "worker": "running pid=3",
                "job_generator": "running pid=4",
            }

        def sighup(self, service):
            self.signaled.append(service)

    mesh_root = tmp_path / "mesh"
    config_path = mesh_root / "auth" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text('{"delay_ms":200}\n', encoding="utf-8")
    baseline = ConfigBaseline.capture(mesh_root)
    config_path.write_text('{"delay_ms":1500}\n', encoding="utf-8")
    manager = SighupProcessManager()
    defender = BlueDefender(
        BlueSelection(
            mode=BlueMode.SCRIPTED,
            level=BlueDefenseLevel.LEVEL_2,
            task_name="phase-2-blue-l2",
        ),
        config_baseline=baseline,
    )

    actions = defender.tick(process_manager=manager, mesh_root=mesh_root)

    assert manager.signaled == ["auth"]
    assert config_path.read_text(encoding="utf-8") == '{"delay_ms":200}\n'
    assert actions == [
        BlueAction(
            kind="config_restore",
            target="auth/config.json",
            status="applied",
            detail="restored baseline config and sent SIGHUP to auth",
        )
    ]


def test_scripted_level_three_removes_malformed_jobs_and_stale_lock(
    tmp_path, monkeypatch
):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:3] == ["redis-cli", "LRANGE", "job_queue"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout='{"id":1}\n{broken\n{"id":2}\n',
                stderr="",
            )
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    defender = BlueDefender(
        BlueSelection(
            mode=BlueMode.SCRIPTED,
            level=BlueDefenseLevel.LEVEL_3,
            task_name="phase-2-blue-l3",
        )
    )

    actions = defender.tick(process_manager=FakeProcessManager(), mesh_root=tmp_path)

    assert ["redis-cli", "DEL", "job_queue"] in calls
    assert ["redis-cli", "RPUSH", "job_queue", '{"id":1}', '{"id":2}'] in calls
    assert ["redis-cli", "DEL", "LOCK:job_processor"] in calls
    assert (
        BlueAction(
            kind="queue_sanitize",
            target="job_queue",
            status="applied",
            detail="removed 1 malformed jobs",
        )
        in actions
    )
    assert (
        BlueAction(
            kind="lock_cleanup",
            target="LOCK:job_processor",
            status="applied",
            detail="deleted stale worker lock",
        )
        in actions
    )


def test_scripted_level_four_triggers_metric_rollback(tmp_path):
    class MetricsPollerWithDrop(FakePoller):
        def get_current_metrics(self):
            return SystemMetrics(
                gateway_success_rate=0.4,
                gateway_p99_latency_ms=1400.0,
                queue_depth=50,
                worker_restart_count=0,
                consumer_stall_count=0,
            )

    baseline_metrics = SystemMetrics(
        gateway_success_rate=1.0,
        gateway_p99_latency_ms=100.0,
        queue_depth=0,
        worker_restart_count=0,
        consumer_stall_count=0,
    )
    defender = BlueDefender(
        BlueSelection(
            mode=BlueMode.SCRIPTED,
            level=BlueDefenseLevel.LEVEL_4,
            task_name="phase-2-blue-l4",
        ),
        baseline_metrics=baseline_metrics,
    )

    actions = defender.tick(
        process_manager=FakeProcessManager(),
        metrics_poller=MetricsPollerWithDrop(),
        mesh_root=tmp_path,
    )

    assert (
        BlueAction(
            kind="metric_rollback",
            target="system",
            status="applied",
            detail="success_rate_drop,latency_spike,queue_growth",
        )
        in actions
    )


def test_llm_showdown_runs_one_defensive_command(tmp_path, monkeypatch):
    captured = {}

    def fake_provider(messages):
        captured["messages"] = messages
        return '{"command":"redis-cli FLUSHALL","reasoning":"clear poisoned state"}'

    def fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout="OK\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    defender = BlueDefender(
        BlueSelection(
            mode=BlueMode.LLM_SHOWDOWN,
            level=BlueDefenseLevel.LEVEL_4,
            task_name="phase-2-blue-llm-showdown",
        ),
        llm_provider=fake_provider,
    )

    actions = defender.tick(
        red_command="redis-cli LPUSH job_queue '{broken'",
        red_reasoning="poison queue",
        process_manager=FakeProcessManager(),
        metrics_poller=FakePoller(),
        project_root=tmp_path,
        mesh_root=tmp_path / "mesh",
    )

    assert captured["command"] == "redis-cli FLUSHALL"
    assert captured["messages"][0]["role"] == "system"
    assert actions == [
        BlueAction(
            kind="llm_command",
            target="redis-cli FLUSHALL",
            status="applied",
            detail="exit_code=0 reasoning=clear poisoned state output=OK",
        )
    ]


def test_blue_prompt_includes_runtime_contract_without_red_reasoning(tmp_path):
    prompt = build_blue_prompt(
        metrics=SystemMetrics(
            gateway_success_rate=0.5,
            gateway_p99_latency_ms=1200.0,
            queue_depth=42,
            worker_restart_count=1,
            consumer_stall_count=0,
        ),
        process_status={"gateway": "running pid=1"},
        red_command="echo '{\"db_write_delay_ms\":5000}' > /mesh/worker/config.json",
        project_root=tmp_path,
        mesh_root=tmp_path / "mesh",
    )

    assert "RED REASONING:" not in prompt
    assert "slow worker writes" not in prompt
    assert "Redis queue key: job_queue" in prompt
    assert "/tmp/gateway.log" in prompt
    assert "worker/config.json" in prompt
    assert "bun run" in prompt
    assert "systemctl is unavailable" in prompt
    assert "/tmp/gateway.pid" in prompt
    assert "restore tampered configs" in prompt


def test_blue_system_prompt_mentions_available_debugging_tools():
    for tool_name in ["netstat", "ss", "lsof"]:
        assert tool_name in BLUE_SYSTEM_PROMPT


def test_blue_system_prompt_allows_proactive_safeguards():
    assert "proactively safeguard" in BLUE_SYSTEM_PROMPT
    assert "monitor" in BLUE_SYSTEM_PROMPT
    assert "harden" in BLUE_SYSTEM_PROMPT
    assert "restart" in BLUE_SYSTEM_PROMPT
    assert "sanitize" in BLUE_SYSTEM_PROMPT
