import json
import subprocess
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Callable, Protocol, cast

from wargames_env.models import SystemMetrics
from wargames_env.server.blue_llm import (
    MetricsProvider,
    ProcessStatusProvider,
    build_default_blue_provider,
    run_blue_llm_tick,
)
from wargames_env.server.config_baseline import ConfigBaseline


class BlueMode(str, Enum):
    SCRIPTED = "scripted"
    LLM_SHOWDOWN = "llm_showdown"


class BlueDefenseLevel(IntEnum):
    LEVEL_0 = 0
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3
    LEVEL_4 = 4


@dataclass(frozen=True)
class BlueSelection:
    mode: BlueMode
    level: BlueDefenseLevel
    task_name: str


@dataclass(frozen=True)
class BlueAction:
    kind: str
    target: str
    status: str
    detail: str = ""

    def model_dump(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "target": self.target,
            "status": self.status,
            "detail": self.detail,
        }


class RestartingProcessManager(ProcessStatusProvider, Protocol):
    def start_all(self) -> None: ...


class ConfigRestoringProcessManager(Protocol):
    def sighup(self, service: str) -> None: ...


@dataclass
class BlueDefender:
    selection: BlueSelection
    config_baseline: ConfigBaseline | None = None
    baseline_metrics: SystemMetrics | None = None
    llm_provider: Callable[[list[dict[str, str]]], str] | None = None
    actions: list[BlueAction] = field(default_factory=list)

    @property
    def mode(self) -> BlueMode:
        return self.selection.mode

    @property
    def level(self) -> BlueDefenseLevel:
        return self.selection.level

    def tick(self, **kwargs: object) -> list[BlueAction]:
        if self.mode == BlueMode.LLM_SHOWDOWN:
            return self._run_llm_showdown(kwargs)

        actions: list[BlueAction] = []
        if self.level >= BlueDefenseLevel.LEVEL_1:
            actions.extend(
                self._restart_stopped_services(
                    cast(RestartingProcessManager, kwargs["process_manager"])
                )
            )
        if self.level >= BlueDefenseLevel.LEVEL_2 and self.config_baseline is not None:
            actions.extend(
                self._restore_modified_configs(
                    cast(ConfigRestoringProcessManager, kwargs["process_manager"])
                )
            )
        if self.level >= BlueDefenseLevel.LEVEL_3:
            actions.extend(self._sanitize_queue_and_lock())
        if self.level >= BlueDefenseLevel.LEVEL_4:
            actions.extend(
                self._rollback_on_metric_drop(
                    cast(MetricsProvider | None, kwargs.get("metrics_poller"))
                )
            )
        return actions

    def _run_llm_showdown(self, kwargs: dict[str, object]) -> list[BlueAction]:
        provider = self.llm_provider or build_default_blue_provider()
        if provider is None:
            return [
                BlueAction(
                    kind="llm_command",
                    target="blue_llm",
                    status="skipped",
                    detail="no Blue LLM API key configured",
                )
            ]

        result = run_blue_llm_tick(
            provider=provider,
            process_manager=cast(ProcessStatusProvider, kwargs["process_manager"]),
            metrics_poller=cast(MetricsProvider, kwargs["metrics_poller"]),
            project_root=cast(Path, kwargs["project_root"]),
            mesh_root=cast(Path, kwargs["mesh_root"]),
            red_command=str(kwargs.get("red_command", "")),
        )
        return [
            BlueAction(
                kind="llm_command",
                target=result.command,
                status=result.status,
                detail=result.detail,
            )
        ]

    def _restart_stopped_services(
        self, process_manager: RestartingProcessManager
    ) -> list[BlueAction]:
        status = process_manager.get_status()
        stopped_services = [
            service for service, value in status.items() if value == "stopped"
        ]
        if not stopped_services:
            return []
        process_manager.start_all()
        return [
            BlueAction(
                kind="restart",
                target=",".join(stopped_services),
                status="applied",
                detail="started stopped services",
            )
        ]

    def _restore_modified_configs(
        self, process_manager: ConfigRestoringProcessManager
    ) -> list[BlueAction]:
        if self.config_baseline is None:
            return []

        actions = []
        restored_paths = self.config_baseline.restore_modified()
        for path in restored_paths:
            service = self.config_baseline.service_for_path(path)
            process_manager.sighup(service)
            relative_path = path.relative_to(self.config_baseline.mesh_root)
            actions.append(
                BlueAction(
                    kind="config_restore",
                    target=relative_path.as_posix(),
                    status="applied",
                    detail=f"restored baseline config and sent SIGHUP to {service}",
                )
            )
        return actions

    def _sanitize_queue_and_lock(self) -> list[BlueAction]:
        result = subprocess.run(
            ["redis-cli", "LRANGE", "job_queue", "0", "-1"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode != 0:
            return []

        raw_jobs = result.stdout.splitlines()
        valid_jobs = []
        malformed_count = 0
        for raw_job in raw_jobs:
            try:
                json.loads(raw_job)
            except json.JSONDecodeError:
                malformed_count += 1
                continue
            valid_jobs.append(raw_job)

        actions = []
        if malformed_count:
            subprocess.run(
                ["redis-cli", "DEL", "job_queue"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if valid_jobs:
                subprocess.run(
                    ["redis-cli", "RPUSH", "job_queue", *valid_jobs],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
            actions.append(
                BlueAction(
                    kind="queue_sanitize",
                    target="job_queue",
                    status="applied",
                    detail=f"removed {malformed_count} malformed jobs",
                )
            )

        subprocess.run(
            ["redis-cli", "DEL", "LOCK:job_processor"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        actions.append(
            BlueAction(
                kind="lock_cleanup",
                target="LOCK:job_processor",
                status="applied",
                detail="deleted stale worker lock",
            )
        )
        return actions

    def _rollback_on_metric_drop(
        self, metrics_poller: MetricsProvider | None
    ) -> list[BlueAction]:
        if metrics_poller is None:
            return []
        metrics_poller.poll_once()
        current_metrics = metrics_poller.get_current_metrics()
        if self.baseline_metrics is None:
            self.baseline_metrics = current_metrics
            return []

        reasons = []
        if (
            self.baseline_metrics.gateway_success_rate
            - current_metrics.gateway_success_rate
            >= 0.2
        ):
            reasons.append("success_rate_drop")
        if (
            current_metrics.gateway_p99_latency_ms
            - self.baseline_metrics.gateway_p99_latency_ms
            >= 500
        ):
            reasons.append("latency_spike")
        if current_metrics.queue_depth - self.baseline_metrics.queue_depth >= 20:
            reasons.append("queue_growth")

        if not reasons:
            return []
        return [
            BlueAction(
                kind="metric_rollback",
                target="system",
                status="applied",
                detail=",".join(reasons),
            )
        ]


def select_blue_defender(task_name: str | None) -> BlueSelection:
    normalized = (task_name or "phase-2-blue-l0").strip()
    if normalized == "phase-2-blue-llm-showdown":
        return BlueSelection(
            mode=BlueMode.LLM_SHOWDOWN,
            level=BlueDefenseLevel.LEVEL_4,
            task_name=normalized,
        )

    prefix = "phase-2-blue-l"
    if normalized.startswith(prefix):
        raw_level = normalized.removeprefix(prefix)
        if raw_level.isdigit():
            level_int = int(raw_level)
            if level_int in [level.value for level in BlueDefenseLevel]:
                return BlueSelection(
                    mode=BlueMode.SCRIPTED,
                    level=BlueDefenseLevel(level_int),
                    task_name=normalized,
                )

    return BlueSelection(
        mode=BlueMode.SCRIPTED,
        level=BlueDefenseLevel.LEVEL_0,
        task_name=normalized,
    )
