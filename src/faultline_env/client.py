from typing import Any

from openenv.core import EnvClient

from .models import (
    StepResult,
    SystemMetrics,
    FaultlineAction,
    FaultlineObservation,
    FaultlineState,
)


class FaultlineClient(EnvClient[FaultlineAction, FaultlineObservation, FaultlineState]):
    def _step_payload(self, action: FaultlineAction) -> dict[str, Any]:
        payload = {"command": action.command}
        if action.reasoning:
            payload["reasoning"] = action.reasoning
        return payload

    def _parse_result(self, payload: dict[str, Any]) -> Any:
        obs_data = payload.get("observation", {})
        metrics_data = obs_data.get("metrics", {})
        reward = float(payload.get("reward", 0.0) or 0.0)
        observation = FaultlineObservation(
            command_output=obs_data.get("command_output", ""),
            metrics=SystemMetrics.model_validate(metrics_data),
            process_status=obs_data.get("process_status", {}),
            done=payload.get("done", False),
            reward=reward,
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=reward,
            done=payload.get("done", False),
            info=payload.get("info", {}),
        )

    def _parse_state(self, payload: dict[str, Any]) -> FaultlineState:
        return FaultlineState.model_validate(payload)
