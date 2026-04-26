#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=0.27",
# ]
# ///
"""Bootstrap the Faultline training repo inside an HF Job container, then exec
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
import time
from pathlib import Path

REPO_URL = os.environ.get(
    "TRAINING_REPO_URL",
    "https://huggingface.co/spaces/Veer15/faultline-env-train",
)
REPO_REF = os.environ.get("TRAINING_REPO_REF", "main")
WORKDIR = Path("/tmp/faultline-train")


def _wait_for_cuda(max_wait_s: int = 180) -> None:
    """Block until CUDA reports a usable device.

    HF Jobs occasionally schedules a container before its NVIDIA driver finishes
    initializing; the first `torch.cuda.is_available()` call then raises
    `Error 802: system not yet initialized`. We poll the device count via a
    short subprocess (so each attempt gets a fresh process and re-tries the
    driver init from scratch) and only proceed once CUDA is healthy.
    """
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
        last_err = (probe.stderr or probe.stdout).strip().splitlines()[-1] if (probe.stderr or probe.stdout).strip() else "unknown"
        print(f"[cuda warmup] attempt {attempt} not ready: {last_err}", flush=True)
        time.sleep(10)
    raise RuntimeError(
        f"CUDA failed to initialize within {max_wait_s}s; last error: {last_err}"
    )


def run(cmd: list[str], **kw) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, **kw)


def main() -> int:
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN missing", file=sys.stderr)
        return 2
    if not os.environ.get("ENV_BASE_URL"):
        print("ENV_BASE_URL missing (point at the live faultline Space)", file=sys.stderr)
        return 2

    if not WORKDIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(WORKDIR)])

    run(["uv", "pip", "install", "--no-cache-dir", "-e", f"{WORKDIR}[training]"])

    _wait_for_cuda(max_wait_s=180)

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
