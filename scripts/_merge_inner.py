"""Inner merge logic, run on a GPU node.

Pulls the adapter from W&B, pushes it to HF Hub, merges into base, pushes the
merged model. Invoked by `scripts/merge_and_push_on_hf.py`.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import wandb

from training.publish.push_adapter import push_adapter
from training.publish.push_merged import push_merged_model


def main() -> None:
    entity = os.environ["WANDB_ENTITY"]
    project = os.environ["WANDB_PROJECT"]
    artifact_name = os.environ["WANDB_ARTIFACT_NAME"]
    alias = os.environ["WANDB_ARTIFACT_ALIAS"]

    api = wandb.Api()
    qualified = f"{entity}/{project}/{artifact_name}:{alias}"
    print(f"[pull] fetching {qualified}", flush=True)
    artifact = api.artifact(qualified, type="model")

    download_root = Path("training/artifacts/wandb_download")
    if download_root.exists():
        shutil.rmtree(download_root)
    download_root.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(artifact.download(root=str(download_root)))
    adapter_dir = _locate_adapter_dir(artifact_dir)
    print(f"[pull] adapter dir: {adapter_dir}", flush=True)

    print("[push] uploading adapter to HF Hub", flush=True)
    push_adapter(folder_path=adapter_dir, log_to_wandb=False)

    print("[merge+push] merging adapter into base and pushing", flush=True)
    push_merged_model(
        adapter_path=adapter_dir,
        output_dir="training/artifacts/merged",
        log_to_wandb=False,
    )
    print("[merge+push] done", flush=True)


def _locate_adapter_dir(root: Path) -> Path:
    for path in root.rglob("adapter_config.json"):
        return path.parent
    raise FileNotFoundError(f"adapter_config.json not found under {root}")


if __name__ == "__main__":
    main()
