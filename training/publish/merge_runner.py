"""Subprocess-safe merge + push of LoRA adapter into base model.

Run as a fresh process from `train_entrypoint.py` after training completes.
Subprocess isolation ensures vLLM/training GPU memory is fully released by the
OS before this loads the base model for merging — in-process vLLM cleanup is
known unreliable (vllm-project/vllm#1908).

Usage:
    python -m training.publish.merge_runner [--adapter-path DIR]

Reads:
    HF_TOKEN (required)
    base_model from training.base.yaml unless --base-model passed
    repo_id from publish.yaml unless --repo-id passed
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter-path",
        default="training/artifacts/checkpoints",
        help="Directory containing adapter_config.json + adapter weights",
    )
    parser.add_argument(
        "--output-dir",
        default="training/artifacts/merged",
        help="Where to write the merged model before upload",
    )
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--training-config", default=os.getenv("TRAINING_CONFIG", "training/config/training.base.yaml"))
    parser.add_argument("--publish-config", default="training/config/publish.yaml")
    args = parser.parse_args()

    if not os.getenv("HF_TOKEN"):
        raise SystemExit("HF_TOKEN required")

    adapter_path = _resolve_adapter_dir(Path(args.adapter_path))
    print(f"[merge] adapter_path={adapter_path}", flush=True)

    base_model = args.base_model
    if base_model is None:
        base_model = yaml.safe_load(Path(args.training_config).read_text())["model"]["base_model"]
    print(f"[merge] base_model={base_model}", flush=True)

    repo_id = args.repo_id or os.getenv("PUBLISH_MERGED_REPO_ID")
    if repo_id is None:
        repo_id = yaml.safe_load(Path(args.publish_config).read_text())["merged_repo_id"]
    print(f"[merge] repo_id={repo_id}", flush=True)

    from training.publish.push_merged import push_merged_model

    push_merged_model(
        base_model=base_model,
        adapter_path=adapter_path,
        output_dir=args.output_dir,
        repo_id=repo_id,
        log_to_wandb=False,
    )
    print("[merge] done", flush=True)
    return 0


def _resolve_adapter_dir(root: Path) -> Path:
    """If `root` itself has adapter_config.json use it; otherwise pick the
    highest-numbered checkpoint subdir."""
    if (root / "adapter_config.json").exists():
        return root
    checkpoints = sorted(
        (p for p in root.glob("checkpoint-*") if (p / "adapter_config.json").exists()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if checkpoints:
        return checkpoints[-1]
    raise FileNotFoundError(
        f"adapter_config.json not found in {root} or any checkpoint-* subdir"
    )


if __name__ == "__main__":
    raise SystemExit(main())
