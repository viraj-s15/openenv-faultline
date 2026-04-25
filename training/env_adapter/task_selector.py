def select_curriculum_tasks(schedule: list[dict], trainer_step: int) -> list[str]:
    for stage in schedule:
        if trainer_step <= int(stage["until_step"]):
            return [str(task) for task in stage["tasks"]]

    if not schedule:
        return []
    return [str(task) for task in schedule[-1]["tasks"]]
