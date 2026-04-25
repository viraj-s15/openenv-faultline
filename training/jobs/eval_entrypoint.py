import os
from pathlib import Path

import yaml

from training.env_adapter.client import WarGamesTrainingClient
from training.grpo.model import load_training_model
from training.grpo.trainer import LocalGenerationClient
from training.rollouts.episode_runner import run_episode
from training.rollouts.transcript_writer import write_transcript


def main() -> None:
    config_path = Path(os.getenv("TRAINING_CONFIG", "training/config/training.base.yaml"))
    settings = yaml.safe_load(config_path.read_text())
    env_client = WarGamesTrainingClient(settings["env"]["base_url"])
    model, tokenizer = load_training_model(settings)
    llm_client = LocalGenerationClient(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=int(settings["trainer"]["max_completion_length"]),
        temperature=float(settings["trainer"]["temperature"]),
    )
    episode = run_episode(
        llm_client=llm_client,
        env_client=env_client,
        task_name="phase-2-blue-l4",
        max_steps=settings["rollout"]["max_steps_per_episode"],
    )
    write_transcript(Path("training/artifacts/eval/latest.json"), episode)


if __name__ == "__main__":
    main()
