from typing import Any

from openenv.core import EnvClient
from openenv.core.env_server.types import State

from .models import StepResult, SystemMetrics, WarGamesAction, WarGamesObservation


class WarGamesClient(EnvClient[WarGamesAction, WarGamesObservation, State]):
    def _step_payload(self, action: WarGamesAction) -> dict[str, Any]:
        return {"command": action.command}

    def _parse_result(self, payload: dict[str, Any]) -> StepResult:
        obs_data = payload.get("observation", {})
        metrics_data = obs_data.get("metrics", {})
        observation = WarGamesObservation(
            command_output=obs_data.get("command_output", ""),
            metrics=SystemMetrics.model_validate(metrics_data),
            process_status=obs_data.get("process_status", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
