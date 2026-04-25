"""Post-training: pull the LoRA adapter from W&B and push it to HF Hub.

Run locally after the GRPO training job completes. This handles the *adapter*
side only — merging into the base Qwen3-8B requires a GPU and is done by
`scripts/merge_and_push_on_hf.py` (an HF Job).

Usage:
    python scripts/pull_and_push_adapter.py \
        --wandb-entity viraj-shah1503-none \
        --wandb-project faultline \
        --artifact-name Qwen3-8B-red-adapter \
        --alias final
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import wandb

from training.publish.push_adapter import push_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", default="faultline")
    parser.add_argument("--artifact-name", default="Qwen3-8B-red-adapter")
    parser.add_argument("--alias", default="final")
    parser.add_argument(
        "--download-dir", default="training/artifacts/wandb_download"
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Override adapter_repo_id from publish.yaml",
    )
    args = parser.parse_args()

    if not os.getenv("HF_TOKEN"):
        raise SystemExit("HF_TOKEN env var required")
    if not os.getenv("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY env var required")

    api = wandb.Api()
    qualified = (
        f"{args.wandb_entity}/{args.wandb_project}/"
        f"{args.artifact_name}:{args.alias}"
    )
    print(f"[pull] fetching {qualified}")
    artifact = api.artifact(qualified, type="model")

    download_root = Path(args.download_dir)
    if download_root.exists():
        shutil.rmtree(download_root)
    download_root.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(artifact.download(root=str(download_root)))
    print(f"[pull] downloaded to {artifact_dir}")

    adapter_dir = _locate_adapter_dir(artifact_dir)
    print(f"[pull] adapter dir resolved to {adapter_dir}")

    print(f"[push] uploading adapter to HF Hub")
    push_adapter(
        repo_id=args.repo_id,
        folder_path=adapter_dir,
        log_to_wandb=False,
    )
    print("[push] done")


def _locate_adapter_dir(root: Path) -> Path:
    """Find the directory containing `adapter_config.json`.

    The W&B artifact may wrap the adapter in a checkpoint subfolder
    (when uploaded by `on_save`) or live at the root (when uploaded by
    `on_train_end` with the trainer output_dir). Walk to find the file.
    """
    for path in root.rglob("adapter_config.json"):
        return path.parent
    raise FileNotFoundError(
        f"adapter_config.json not found under {root}; "
        f"contents: {list(root.rglob('*'))[:20]}"
    )


if __name__ == "__main__":
    main()
