import json
from pathlib import Path

from training.rollouts.trajectory import EpisodeTrajectory


def write_transcript(path: Path, episode: EpisodeTrajectory) -> None:
    payload = {
        "task_name": episode.task_name,
        "rewards": episode.rewards,
        "steps": [step.__dict__ for step in episode.steps],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
