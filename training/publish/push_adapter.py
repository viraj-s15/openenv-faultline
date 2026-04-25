"""Push a LoRA adapter to the HuggingFace Hub with a generated model card."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from huggingface_hub import HfApi

from training.publish.model_card import build_model_card_text


def _load_publish_config(path: str | Path | None) -> dict:
    if path is None:
        path = Path("training/config/publish.yaml")
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def push_adapter(
    repo_id: str | None = None,
    folder_path: str | Path = "training/artifacts/checkpoints",
    private: bool | None = None,
    base_model: str | None = None,
    license: str | None = None,
    config_path: str | Path | None = None,
    log_to_wandb: bool = True,
):
    """Upload the LoRA adapter directory at `folder_path` to `repo_id`.

    Defaults are sourced from `training/config/publish.yaml` and
    `training/config/training.base.yaml` so callers can pass none.
    """
    publish_cfg = _load_publish_config(config_path)
    repo_id = repo_id or publish_cfg["adapter_repo_id"]
    private = publish_cfg.get("private", False) if private is None else private
    license = license or publish_cfg.get("license", "mit")
    if base_model is None:
        training_yaml = Path("training/config/training.base.yaml")
        if training_yaml.exists():
            base_model = yaml.safe_load(training_yaml.read_text())["model"]["base_model"]

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"adapter folder not found: {folder}")

    card_text = build_model_card_text(
        repo_id=repo_id,
        base_model=base_model or "unknown",
        artifact_kind="adapter",
        license=license,
    )
    (folder / "README.md").write_text(card_text, encoding="utf-8")

    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    result = api.upload_folder(repo_id=repo_id, folder_path=str(folder))

    if log_to_wandb:
        _log_wandb_artifact(folder, name=f"{Path(repo_id).name}-adapter")

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
