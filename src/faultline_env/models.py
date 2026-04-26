from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, field_validator


class SystemMetrics(BaseModel):
    gateway_success_rate: float = Field(..., ge=0.0, le=1.0)
    gateway_p99_latency_ms: float = Field(..., ge=0.0)
    queue_depth: int = Field(..., ge=0)
    worker_restart_count: int = Field(..., ge=0)
    consumer_stall_count: int = Field(..., ge=0)


class FaultlineAction(Action):
    command: str = Field(..., description="Single bash command to execute")
    reasoning: str | None = Field(
        default=None, description="Optional concise reason for the command"
    )

    @field_validator("command")
    @classmethod
    def command_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("command must not be empty")
        return value

    @field_validator("reasoning")
    @classmethod
    def empty_reasoning_as_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class FaultlineObservation(Observation):
    command_output: str = Field(
        ..., description="stdout and stderr from the last executed command"
    )
    metrics: SystemMetrics
    process_status: dict[str, str] = Field(default_factory=dict)


class FaultlineState(BaseModel):
    episode_id: str | None = None
    task_name: str
    step_count: int = Field(..., ge=0)
    max_steps: int = Field(..., ge=1)
    blue_mode: str
    blue_level: int = Field(..., ge=0)
    metrics: SystemMetrics
    process_status: dict[str, str] = Field(default_factory=dict)


class StepResult(BaseModel):
    observation: FaultlineObservation
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
