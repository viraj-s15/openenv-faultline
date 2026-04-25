import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from wargames_env.models import StepResult, WarGamesAction, WarGamesObservation
from wargames_env.server.blue_defender import BlueDefender, select_blue_defender
from wargames_env.server.config_baseline import ConfigBaseline
from wargames_env.server.metrics_poller import MetricsPoller
from wargames_env.server.process_manager import ProcessManager


@dataclass(frozen=True)
class RedCommandResult:
    command: str
    output: str
    exit_code: int
    timed_out: bool
    duration_ms: int


class WarGamesEnv:
    def __init__(
        self, project_root: Path | None = None, mesh_root: Path | None = None
    ) -> None:
        default_root = Path(__file__).resolve().parents[3]
        self.project_root = (project_root or default_root).resolve()
        self.mesh_root = (
            mesh_root or Path(os.getenv("MESH_ROOT", self.project_root / "mesh"))
        ).resolve()
        self.episode_id = str(uuid4())
        self.step_count = 0
        self._process_manager = ProcessManager(
            project_root=self.project_root, mesh_root=self.mesh_root
        )
        self._metrics_poller = MetricsPoller(poll_interval_s=2.0)
        self.max_steps = 10
        self.last_exit_code = 0
        self._blue_defender = BlueDefender(select_blue_defender(None))

    def _write_default_registry(self) -> None:
        self.mesh_root.mkdir(parents=True, exist_ok=True)
        registry_path = self.mesh_root / "registry.json"
        registry_path.write_text(
            '{"services":{"auth":{"host":"localhost","port":3001,"protocol":"http"},'
            '"redis":{"host":"localhost","port":6379,"protocol":"tcp"},'
            '"worker":{"host":"localhost","port":null,"protocol":"internal"}}}\n',
            encoding="utf-8",
        )

    def _reset_runtime_files(self) -> None:
        for log_name in ["gateway", "auth", "worker", "job_gen"]:
            Path(f"/tmp/{log_name}.log").write_text("", encoding="utf-8")
        Path("/tmp/worker_restart_count").write_text("0", encoding="utf-8")
        Path("/tmp/consumer_stall_count").write_text("0", encoding="utf-8")

    def _redis_flush(self) -> None:
        subprocess.run(
            ["redis-cli", "FLUSHDB"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

    def _observation(
        self, command_output: str, done: bool, reward: float
    ) -> WarGamesObservation:
        self._metrics_poller.poll_once()
        metrics = self._metrics_poller.get_current_metrics()
        return WarGamesObservation(
            command_output=command_output,
            metrics=metrics,
            process_status=self._process_manager.get_status(),
            done=done,
            reward=reward,
        )

    def _run_red_command(self, command: str, timeout_s: float) -> RedCommandResult:
        started_at = time.monotonic()
        shell_path = ":".join(
            path
            for path in [
                os.environ.get("PATH", ""),
                "/usr/local/sbin",
                "/usr/local/bin",
                "/usr/sbin",
                "/usr/bin",
                "/sbin",
                "/bin",
            ]
            if path
        )
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd="/",
                env={
                    **os.environ,
                    "APP_ROOT": str(self.project_root),
                    "MESH_ROOT": str(self.mesh_root),
                    "PATH": shell_path,
                },
                check=False,
            )
            exit_code = result.returncode
            output = (result.stdout + result.stderr).strip() or "(no output)"
            timed_out = False
        except subprocess.TimeoutExpired:
            exit_code = 124
            output = f"Command timed out after {timeout_s:g} seconds."
            timed_out = True

        duration_ms = max(0, int((time.monotonic() - started_at) * 1000))
        return RedCommandResult(
            command=command,
            output=output,
            exit_code=exit_code,
            timed_out=timed_out,
            duration_ms=duration_ms,
        )

    def reset(
        self,
        task_name: str | None = None,
        **kwargs: object,
    ) -> WarGamesObservation:
        self.episode_id = str(uuid4())
        self.step_count = 0
        self._blue_defender = BlueDefender(select_blue_defender(task_name))
        Path("/tmp/current_task").write_text(
            self._blue_defender.selection.task_name, encoding="utf-8"
        )
        self._write_default_registry()
        self._blue_defender.config_baseline = ConfigBaseline.capture(self.mesh_root)
        self._reset_runtime_files()
        self._redis_flush()
        self._process_manager.restart_all()
        if not self._process_manager.wait_healthy(timeout_s=30):
            raise RuntimeError("Services failed health checks after reset")
        self._metrics_poller.poll_once()
        self._blue_defender.baseline_metrics = (
            self._metrics_poller.get_current_metrics()
        )
        return self._observation("WarGames mesh ready.", done=False, reward=0.0)

    def step(
        self,
        action: WarGamesAction,
        timeout_s: float | None = None,
        **kwargs: object,
    ) -> StepResult:
        self.step_count += 1
        timeout = timeout_s or 10
        command_result = self._run_red_command(action.command, timeout_s=timeout)
        self.last_exit_code = command_result.exit_code
        blue_actions = self._blue_defender.tick(
            red_command=command_result.command,
            red_exit_code=command_result.exit_code,
            process_manager=self._process_manager,
            metrics_poller=self._metrics_poller,
            project_root=self.project_root,
            mesh_root=self.mesh_root,
        )
        done = self.step_count >= self.max_steps
        observation = self._observation(command_result.output, done=done, reward=0.0)
        return StepResult(
            observation=observation,
            reward=0.0,
            done=done,
            info={
                "exit_code": command_result.exit_code,
                "timed_out": command_result.timed_out,
                "command": command_result.command,
                "duration_ms": command_result.duration_ms,
                "blue_actions": [
                    blue_action.model_dump() for blue_action in blue_actions
                ],
            },
        )

    def state(self) -> dict[str, object]:
        self._metrics_poller.poll_once()
        metrics = self._metrics_poller.get_current_metrics()
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "blue_mode": self._blue_defender.mode.value,
            "blue_level": int(self._blue_defender.level),
            "metrics": metrics.model_dump(),
            "process_status": self._process_manager.get_status(),
        }

    def close(self) -> None:
        self._metrics_poller.stop()
