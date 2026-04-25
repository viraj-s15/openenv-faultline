from dataclasses import dataclass

try:
    from trl import GRPOConfig as TrlGRPOConfig
except ModuleNotFoundError:  # pragma: no cover - exercised when training extras are absent
    TrlGRPOConfig = None


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


def build_grpo_config(settings: dict):
    trainer = settings["trainer"]
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
        "report_to": "none",
    }
    if TrlGRPOConfig is not None:
        return TrlGRPOConfig(**config_kwargs)
    return LocalGRPOConfig(**config_kwargs)
