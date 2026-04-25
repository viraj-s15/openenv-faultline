#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=0.27",
# ]
# ///
"""Bootstrap the WarGames training repo inside an HF Job container, then exec
the GRPO training entrypoint.

Job constraints:
  - HF Jobs uv-run takes a single script file; the rest of the package isn't
    auto-uploaded. We clone the Space repo (public, includes training/) and
    pip-install it with the [training] extra inside the same container.
  - All required env (HF_TOKEN, WANDB_API_KEY, ENV_BASE_URL, TRAINING_CONFIG)
    must be passed via `--secrets` / `--env` on `hf jobs uv run`.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = os.environ.get(
    "TRAINING_REPO_URL",
    "https://huggingface.co/spaces/Veer15/wargames-env-train",
)
REPO_REF = os.environ.get("TRAINING_REPO_REF", "main")
WORKDIR = Path("/tmp/wargames-train")


def run(cmd: list[str], **kw) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, **kw)


def main() -> int:
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN missing", file=sys.stderr)
        return 2
    if not os.environ.get("ENV_BASE_URL"):
        print("ENV_BASE_URL missing (point at the live wargames Space)", file=sys.stderr)
        return 2

    if not WORKDIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(WORKDIR)])

    run([
        sys.executable, "-m", "pip", "install", "--no-cache-dir",
        "-e", f"{WORKDIR}[training]",
    ])

    env = os.environ.copy()
    env.setdefault("TRAINING_CONFIG", "training/config/training.smoke.yaml")
    run(
        [sys.executable, "training/jobs/train_entrypoint.py"],
        cwd=str(WORKDIR),
        env=env,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
