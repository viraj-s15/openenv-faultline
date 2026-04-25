"""Write episode transcripts to disk as JSON."""

from __future__ import annotations

import json
from pathlib import Path

from training.rollouts.trajectory import EpisodeTrajectory


def _json_default(value):
    """Best-effort fallback for non-serializable info values."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return repr(value)


def write_transcript(path: Path, episode: EpisodeTrajectory) -> None:
    payload = {
        "task_name": episode.task_name,
        "rewards": episode.rewards,
        "steps": [step.__dict__ for step in episode.steps],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
