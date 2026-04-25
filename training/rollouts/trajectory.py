from dataclasses import dataclass
from typing import Any


@dataclass
class RolloutStep:
    step_num: int
    prompt: str
    raw_completion: str
    command: str
    reward: float
    done: bool
    info: dict[str, Any]


@dataclass
class EpisodeTrajectory:
    task_name: str
    steps: list[RolloutStep]
    rewards: list[float]
