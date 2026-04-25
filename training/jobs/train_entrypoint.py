"""HuggingFace Jobs entrypoint for GRPO training of the WarGames Red agent.

Wires together:
  - Live env client (FastAPI WarGames server)
  - unsloth + LoRA model load
  - TRL >=0.25 GRPOTrainer with a real rollout_func + reward_func pair
  - Curriculum callback (hard-switch by trainer step)
  - W&B artifact callback (adapter at every checkpoint + final)

Required env:
  - HF_TOKEN: HuggingFace API token
  - WANDB_API_KEY: required when wandb.enabled is true in the YAML
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import yaml

from training.env_adapter.client import WarGamesTrainingClient
from training.env_adapter.task_selector import select_curriculum_tasks
from training.grpo.callbacks import CurriculumCallback, WandbArtifactCallback
from training.grpo.config import configure_wandb
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

    wandb_enabled = bool(settings.get("wandb", {}).get("enabled", False))
    if wandb_enabled and not os.getenv("WANDB_API_KEY"):
        raise RuntimeError(
            "wandb.enabled=true in config but WANDB_API_KEY is not set; "
            "pass it via `hf jobs ... --secrets WANDB_API_KEY HF_TOKEN ...`"
        )

    configure_wandb(settings)

    env_client = WarGamesTrainingClient(settings["env"]["base_url"])
    model, tokenizer = load_training_model(settings)
    llm_client = LocalGenerationClient(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=int(settings["trainer"]["max_completion_length"]),
        temperature=float(settings["trainer"]["temperature"]),
    )

    schedule = curriculum["schedule"]
    initial_tasks = select_curriculum_tasks(schedule, trainer_step=0)

    def dataset_builder(tasks: Sequence[str]):
        return build_prompt_dataset(tasks=list(tasks), env_client=env_client)

    trainer_ref: dict = {"trainer": None}

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset_builder(initial_tasks),
        reward_funcs=[reward_from_rollout],
        rollout_func=make_rollout_func(
            llm_client=llm_client,
            env_client=env_client,
            max_steps=settings["rollout"]["max_steps_per_episode"],
            tokenizer=tokenizer,
        ),
        settings=settings,
    )
    trainer_ref["trainer"] = trainer

    trainer.add_callback(
        CurriculumCallback(
            schedule=schedule,
            dataset_builder=dataset_builder,
            trainer_ref=lambda: trainer_ref["trainer"],
            initial_step=0,
        )
    )
    if wandb_enabled:
        trainer.add_callback(
            WandbArtifactCallback(
                artifact_name=f"{settings['model']['base_model'].split('/')[-1]}-red-adapter",
            )
        )

    trainer.train()


if __name__ == "__main__":
    main()
