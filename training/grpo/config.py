"""Map repository YAML settings onto TRL `GRPOConfig`."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is an optional dep
    wandb = None

try:
    from trl import GRPOConfig as TrlGRPOConfig
except ModuleNotFoundError:  # pragma: no cover - exercised when training extras are absent
    TrlGRPOConfig = None


def _resolve_wandb_run_name(settings: dict) -> str:
    """Generate a descriptive run name from config when none is set."""
    base_model = settings.get("model", {}).get("base_model", "model")
    base = base_model.split("/")[-1]
    lr = settings["trainer"]["learning_rate"]
    return f"{base}-lr{lr}"


def configure_wandb(settings: dict) -> None:
    """Set WANDB_* env vars when wandb is enabled.

    Caller responsibilities (validated in train_entrypoint):
      - When `wandb.enabled: true`, `WANDB_API_KEY` MUST be present.
    """
    wandb_cfg = settings.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        os.environ["WANDB_MODE"] = "disabled"
        return

    if wandb is None:
        print("[wandb] enabled in config but `wandb` not installed - disabling")
        os.environ["WANDB_MODE"] = "disabled"
        return

    project = wandb_cfg.get("project") or "faultline"
    entity = wandb_cfg.get("entity") or None
    run_name = wandb_cfg.get("run_name") or _resolve_wandb_run_name(settings)

    # Force online mode when explicitly enabled so jobs do not silently fall
    # back to offline/disabled because of an outer env override.
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_PROJECT"] = project
    if entity:
        os.environ["WANDB_ENTITY"] = entity
    os.environ["WANDB_NAME"] = run_name


@dataclass
class LocalGRPOConfig:
    """Fallback config dataclass when TRL is not installed (tests, lint)."""

    output_dir: str
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_generations: int
    max_completion_length: int
    temperature: float
    beta: float
    use_vllm: bool
    report_to: str
    run_name: str | None = None
    max_steps: int = -1
    num_train_epochs: float = 3.0
    save_steps: int = 500
    logging_steps: int = 10
    log_completions: bool = False


def build_grpo_config(settings: dict):
    trainer = settings["trainer"]
    wandb_cfg = settings.get("wandb", {})
    wandb_enabled = wandb_cfg.get("enabled", False)

    config_kwargs: dict = {
        "output_dir": trainer["output_dir"],
        "learning_rate": trainer["learning_rate"],
        "per_device_train_batch_size": trainer["per_device_train_batch_size"],
        "gradient_accumulation_steps": trainer["gradient_accumulation_steps"],
        "num_generations": trainer["num_generations"],
        "max_completion_length": trainer["max_completion_length"],
        "temperature": trainer["temperature"],
        "beta": trainer["beta"],
        "use_vllm": trainer["use_vllm"],
        "report_to": "wandb" if wandb_enabled else "none",
        "run_name": wandb_cfg.get("run_name") or _resolve_wandb_run_name(settings),
        # Surface Red completions as W&B tables so per-step bash commands are
        # visible alongside the loss/reward scalars.
        "log_completions": bool(wandb_enabled),
    }
    # Optional schedule controls; leave TRL defaults intact when absent.
    for key in (
        "max_steps",
        "num_train_epochs",
        "save_steps",
        "logging_steps",
        "vllm_mode",
        "vllm_gpu_memory_utilization",
        "vllm_tensor_parallel_size",
        "gradient_checkpointing",
    ):
        if key in trainer:
            config_kwargs[key] = trainer[key]

    if TrlGRPOConfig is not None:
        return TrlGRPOConfig(**config_kwargs)
    return LocalGRPOConfig(**config_kwargs)
