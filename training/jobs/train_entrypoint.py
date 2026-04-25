import os
from pathlib import Path

import yaml

from training.env_adapter.client import WarGamesTrainingClient
from training.env_adapter.task_selector import select_curriculum_tasks
from training.grpo.model import load_training_model
from training.grpo.trainer import (
    LocalGenerationClient,
    build_prompt_dataset,
    build_trainer,
    make_rollout_func,
    reward_from_rollout,
)


def main() -> None:
    config_path = Path(os.getenv("TRAINING_CONFIG", "training/config/training.base.yaml"))
    curriculum_path = Path(
        os.getenv("CURRICULUM_CONFIG", "training/config/curriculum.l0-l4.yaml")
    )
    settings = yaml.safe_load(config_path.read_text())
    curriculum = yaml.safe_load(curriculum_path.read_text())
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required for Hugging Face Jobs")

    env_client = WarGamesTrainingClient(settings["env"]["base_url"])
    model, tokenizer = load_training_model(settings)
    llm_client = LocalGenerationClient(model=model, tokenizer=tokenizer)
    tasks = select_curriculum_tasks(curriculum["schedule"], trainer_step=0)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=build_prompt_dataset(tasks),
        reward_funcs=[reward_from_rollout],
        rollout_func=make_rollout_func(
            llm_client=llm_client,
            env_client=env_client,
            max_steps=settings["rollout"]["max_steps_per_episode"],
        ),
        settings=settings,
    )
    trainer.train()


if __name__ == "__main__":
    main()
