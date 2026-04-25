from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    name: str
    max_steps: int


TASK_CONFIGS = {
    "phase-0-healthy-mesh": TaskConfig("phase-0-healthy-mesh", 10),
    "phase-1-raw-bash-red": TaskConfig("phase-1-raw-bash-red", 10),
    "phase-2-blue-l0": TaskConfig("phase-2-blue-l0", 10),
    "phase-2-blue-l1": TaskConfig("phase-2-blue-l1", 10),
    "phase-2-blue-l2": TaskConfig("phase-2-blue-l2", 10),
    "phase-2-blue-l3": TaskConfig("phase-2-blue-l3", 10),
    "phase-2-blue-l4": TaskConfig("phase-2-blue-l4", 10),
    "phase-2-blue-llm-showdown": TaskConfig("phase-2-blue-llm-showdown", 10),
}

DEFAULT_TASK_NAME = "phase-2-blue-l0"


def get_task_config(task_name: str | None) -> TaskConfig:
    normalized = (task_name or DEFAULT_TASK_NAME).strip()
    return TASK_CONFIGS.get(normalized, TaskConfig(normalized, 10))
