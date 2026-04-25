"""HF Jobs entrypoint: pull adapter from W&B, merge into base, push merged model.

Designed to run via `hf jobs uv run` on a GPU instance (a100-large or similar).
Reads:
    HF_TOKEN, WANDB_API_KEY (secrets)
    WANDB_ENTITY, WANDB_PROJECT, WANDB_ARTIFACT_NAME, WANDB_ARTIFACT_ALIAS (env)

Bootstrap mirrors `training/jobs/run_on_hf.py`: clones the repo Space, pip
installs, then execs this file.
"""
# /// script
# dependencies = [
#   "huggingface_hub>=0.27",
# ]
# ///

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = os.environ.get(
    "TRAINING_REPO_URL",
    "https://huggingface.co/spaces/Veer15/wargames-env-train",
)
REPO_REF = os.environ.get("TRAINING_REPO_REF", "main")
WORKDIR = Path("/tmp/wargames-train")


def _wait_for_cuda(max_wait_s: int = 180) -> None:
    deadline = time.monotonic() + max_wait_s
    attempt = 0
    last_err: str | None = None
    while time.monotonic() < deadline:
        attempt += 1
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import torch; "
                    "assert torch.cuda.is_available(), 'no cuda'; "
                    "n = torch.cuda.device_count(); "
                    "assert n > 0, f'device_count={n}'; "
                    "_ = torch.cuda.get_device_name(0); "
                    "print(f'CUDA ok: {n} device(s), {torch.cuda.get_device_name(0)}')"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if probe.returncode == 0:
            print(probe.stdout.strip(), flush=True)
            return
        last_err = (
            (probe.stderr or probe.stdout).strip().splitlines()[-1]
            if (probe.stderr or probe.stdout).strip()
            else "unknown"
        )
        print(f"[cuda warmup] attempt {attempt} not ready: {last_err}", flush=True)
        time.sleep(10)
    raise RuntimeError(
        f"CUDA failed to initialize within {max_wait_s}s; last error: {last_err}"
    )


def run(cmd: list[str], **kw) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, **kw)


def main() -> int:
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN required")
    if not os.getenv("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY required")

    if not WORKDIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(WORKDIR)])

    run(["uv", "pip", "install", "--no-cache-dir", "-e", f"{WORKDIR}[training]"])
    run(["uv", "pip", "install", "--no-cache-dir", "wandb"])

    _wait_for_cuda(max_wait_s=180)

    env = os.environ.copy()
    env.setdefault("WANDB_ENTITY", "viraj-shah1503-none")
    env.setdefault("WANDB_PROJECT", "faultline")
    env.setdefault("WANDB_ARTIFACT_NAME", "Qwen3-8B-red-adapter")
    env.setdefault("WANDB_ARTIFACT_ALIAS", "final")

    script = WORKDIR / "scripts" / "_merge_inner.py"
    run([sys.executable, str(script)], cwd=str(WORKDIR), env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
