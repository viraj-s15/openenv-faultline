import os
from dataclasses import dataclass

try:
    import wandb
except ImportError:
    wandb = None

try:
    from trl import GRPOConfig as TrlGRPOConfig
except ModuleNotFoundError:  # pragma: no cover - exercised when training extras are absent
    TrlGRPOConfig = None


def _resolve_wandb_run_name(settings: dict) -> str:
    """Generate a descriptive run name from config if not explicitly set."""
    base = settings["model"]["base_model"].split("/")[-1]
    lr = settings["trainer"]["learning_rate"]
    return f"{base}-lr{lr}"


def configure_wandb(settings: dict) -> None:
    """Set WANDB_* env vars and log in if wandb is enabled and installed."""
    wandb_cfg = settings.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        os.environ["WANDB_MODE"] = "disabled"
        return

    if wandb is None:
        print("[wandb] enabled in config but `wandb` not installed — skipping")
        os.environ["WANDB_MODE"] = "disabled"
        return

    project = wandb_cfg.get("project") or "faultline"
    entity = wandb_cfg.get("entity") or None
    run_name = wandb_cfg.get("run_name") or _resolve_wandb_run_name(settings)

    os.environ.setdefault("WANDB_PROJECT", project)
    if entity:
        os.environ.setdefault("WANDB_ENTITY", entity)
    os.environ.setdefault("WANDB_NAME", run_name)

    # wandb.init is called by TRL automatically when report_to="wandb"


@dataclass
class LocalGRPOConfig:
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


def build_grpo_config(settings: dict):
    trainer = settings["trainer"]
    wandb_cfg = settings.get("wandb", {})
    wandb_enabled = wandb_cfg.get("enabled", False)

    config_kwargs = {
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
    }
    if TrlGRPOConfig is not None:
        return TrlGRPOConfig(**config_kwargs)
    return LocalGRPOConfig(**config_kwargs)