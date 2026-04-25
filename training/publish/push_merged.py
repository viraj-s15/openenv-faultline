"""Merge a LoRA adapter into the base model and push the merged model to the Hub."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from training.publish.model_card import build_model_card_text


def _load_publish_config(path: str | Path | None) -> dict:
    if path is None:
        path = Path("training/config/publish.yaml")
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def export_merged_model(
    base_model: str,
    adapter_path: str | Path,
    output_dir: str | Path,
    torch_dtype: Any = None,
    device_map: str = "auto",
) -> str:
    """Merge `adapter_path` into `base_model` and save the result to `output_dir`.

    Defaults are tuned for big models on a single accelerator: bf16 + auto device
    map + low CPU mem usage so a 9B base does not blow CPU RAM during merge.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    output_dir = str(output_dir)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(model, str(adapter_path))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(output_dir)
    return output_dir


def push_merged_model(
    base_model: str | None = None,
    adapter_path: str | Path = "training/artifacts/checkpoints",
    output_dir: str | Path = "training/artifacts/merged",
    repo_id: str | None = None,
    private: bool | None = None,
    license: str | None = None,
    config_path: str | Path | None = None,
    log_to_wandb: bool = True,
):
    """Merge adapter into base, write a model card, push the merged repo to the Hub."""
    publish_cfg = _load_publish_config(config_path)
    repo_id = repo_id or publish_cfg["merged_repo_id"]
    private = publish_cfg.get("private", False) if private is None else private
    license = license or publish_cfg.get("license", "mit")

    if base_model is None:
        training_yaml = Path("training/config/training.base.yaml")
        base_model = yaml.safe_load(training_yaml.read_text())["model"]["base_model"]

    merged_dir = Path(
        export_merged_model(
            base_model=base_model,
            adapter_path=adapter_path,
            output_dir=output_dir,
        )
    )

    card_text = build_model_card_text(
        repo_id=repo_id,
        base_model=base_model,
        artifact_kind="merged",
        license=license,
    )
    (merged_dir / "README.md").write_text(card_text, encoding="utf-8")

    from huggingface_hub import HfApi

    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    result = api.upload_folder(repo_id=repo_id, folder_path=str(merged_dir))

    if log_to_wandb:
        _log_wandb_artifact(merged_dir, name=f"{Path(repo_id).name}-merged")

    return result


def _log_wandb_artifact(folder: Path, name: str) -> None:
    try:
        import wandb
    except ImportError:
        return
    if os.environ.get("WANDB_MODE", "online") == "disabled":
        return
    if wandb.run is None:
        return
    artifact = wandb.Artifact(name, type="model")
    artifact.add_dir(str(folder))
    wandb.log_artifact(artifact, aliases=["pushed"])
