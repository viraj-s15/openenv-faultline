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

    env_base_url = os.getenv("ENV_BASE_URL")
    if env_base_url:
        settings["env"]["base_url"] = env_base_url
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
    model, tokenizer, lora_config = load_training_model(settings)

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
            env_client=env_client,
            max_steps=settings["rollout"]["max_steps_per_episode"],
            tokenizer=tokenizer,
        ),
        settings=settings,
        peft_config=lora_config,
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

    if os.getenv("PUBLISH_ON_FINISH", "true").lower() in {"1", "true", "yes"}:
        _publish_artifacts(settings, trainer)


def _publish_artifacts(settings: dict, trainer) -> None:
    """After training: save final adapter, push it to HF Hub, then merge+push
    in a subprocess so vLLM's GPU memory is released before the base model is
    loaded for merging (in-process vLLM cleanup is unreliable)."""
    import subprocess as _sub
    import sys as _sys
    from pathlib import Path as _Path

    output_dir = _Path(settings["trainer"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[publish] saving final adapter to {output_dir}", flush=True)
    try:
        trainer.save_model(str(output_dir))
    except Exception as exc:
        print(f"[publish] trainer.save_model failed: {exc}", flush=True)

    if not (output_dir / "adapter_config.json").exists():
        print(f"[publish] no adapter_config.json under {output_dir}; skipping push", flush=True)
        return

    from training.publish.push_adapter import push_adapter

    print("[publish] pushing adapter to HF Hub", flush=True)
    try:
        push_adapter(folder_path=output_dir, log_to_wandb=False)
        print("[publish] adapter push complete", flush=True)
    except Exception as exc:
        print(f"[publish] adapter push failed: {exc}", flush=True)

    if os.getenv("PUBLISH_MERGED", "true").lower() not in {"1", "true", "yes"}:
        return

    print("[publish] launching merge subprocess (clean GPU context)", flush=True)
    proc = _sub.run(
        [_sys.executable, "-m", "training.publish.merge_runner",
         "--adapter-path", str(output_dir)],
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        print(f"[publish] merge subprocess exited {proc.returncode}", flush=True)
    else:
        print("[publish] merged push complete", flush=True)


if __name__ == "__main__":
    main()
