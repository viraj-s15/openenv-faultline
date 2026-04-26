# Running GRPO Training

This document covers training the Faultline Red agent with GRPO on Hugging Face Jobs against a live environment Space.

## Architecture

```
training/jobs/run_on_hf.py        ← bootstrap that clones the env Space repo and execs the trainer
training/jobs/train_entrypoint.py ← TRL GRPOTrainer + curriculum + W&B + publish hook
training/grpo/                    ← trainer wiring, model loading, reward fns
training/env_adapter/             ← retry-resilient HTTP client to the env Space
training/publish/                 ← LoRA + merged push at end of run
training/config/                  ← YAML configs (base, smoke, curriculum, publish, space)
```

## Required Environment Variables

Training reads three secrets and one URL:

```bash
HF_TOKEN=hf_...           # write access to publish target repos
WANDB_API_KEY=...         # required when wandb.enabled: true (default)
ENV_BASE_URL=https://your-namespace-faultline-env-train.hf.space
```

Store them in `.secrets` at the repo root (keys + values, one per line):

```text
HF_TOKEN=hf_...
WANDB_API_KEY=...
```

`hf jobs uv run --secrets-file ./.secrets` injects all of them. The bare-name form (`--secrets WANDB_API_KEY`) only works for `HF_TOKEN`.

## Configure Training

Edit `training/config/training.base.yaml` for the real run and `training/config/training.smoke.yaml` for the cheap shape check. Key knobs:

| Key | Base default | What it does |
|---|---|---|
| `model.base_model` | `Qwen/Qwen3-8B` | Base model |
| `model.lora_rank` | 16 | LoRA rank |
| `trainer.learning_rate` | 5e-6 | GRPO LR |
| `trainer.num_generations` | 4 | Group size for GRPO advantages |
| `trainer.max_completion_length` | 1536 | Tokens per completion |
| `trainer.use_vllm` | true | Required so `rollout_func` actually fires |
| `trainer.vllm_gpu_memory_utilization` | 0.65 | Lower this if backward OOMs |
| `trainer.max_steps` | 60 | Total training steps |
| `trainer.save_steps` | 50 | Checkpoint cadence |
| `rollout.max_steps_per_episode` | 4 | Env steps per generation |
| `rollout.reward_aggregation` | sum | How step rewards combine into episode reward |

Curriculum lives in `training/config/curriculum.l0-l4.yaml`. Override via `CURRICULUM_CONFIG=path/to/file.yaml`.

## Smoke Run (a10g-large, ~$1.50)

Validates model loading, rollout, parser, and the publish path end-to-end on a small model.

```bash
hf jobs uv run --detach \
  --flavor a10g-large --timeout 1h \
  --secrets-file ./.secrets \
  --env "TRAINING_CONFIG=training/config/training.smoke.yaml" \
  --env "ENV_BASE_URL=https://your-namespace-faultline-env-train.hf.space" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  --env "PUBLISH_ON_FINISH=true" \
  --env "PUBLISH_MERGED=true" \
  training/jobs/run_on_hf.py
```

Smoke pushes to `faultline-red-smoke-{lora,merged}` (defaults in `training/config/publish.yaml`).

## Real Run (h200, ~$13 for 60 steps)

```bash
hf jobs uv run --detach \
  --flavor h200 --timeout 4h \
  --secrets-file ./.secrets \
  --env "TRAINING_CONFIG=training/config/training.base.yaml" \
  --env "ENV_BASE_URL=https://your-namespace-faultline-env-train.hf.space" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  --env "PUBLISH_ON_FINISH=true" \
  --env "PUBLISH_MERGED=true" \
  training/jobs/run_on_hf.py
```

Empirical step time on h200 with Qwen3-8B + 4-step episodes is ~150s/step. Budget accordingly:

| `max_steps` | Wall clock | Cost @ $5/h |
|---|---|---|
| 60 | ~2h30m | ~$13 |
| 80 | ~3h20m | ~$17 |
| 100 | ~4h10m | ~$22 |

## Job Lifecycle

Inspect:

```bash
hf jobs ps -a | head
hf jobs inspect <JOB_ID>
hf jobs logs <JOB_ID> | tail -100
```

Cancel:

```bash
hf jobs cancel <JOB_ID>
```

The W&B run URL is printed in the first ~90s of logs once `wandb.login()` completes. Bookmark it; it's the only readable view of progress (HF Jobs logs render the per-step prompts table verbatim and become unreadable mid-run).

## What to Watch in W&B

GRPO is a policy-gradient method. Loss does not monotonically decrease — ignore it as a progress metric.

Useful metrics, ranked:

1. `train/reward` (mean) and per-fn means (`reward_from_rollout/mean`, `reward_parse_success/mean`) — should trend up
2. `train/reward_std` — must stay > 0; if it goes flat, GRPO has no advantages and learning stalls
3. `train/frac_reward_zero_std` — fraction of groups where all generations got equal reward; should stay near 0
4. `train/completions/mean_length` — watch for collapse (smoke run dropped to 13 tokens once)
5. `train/kl` — divergence from the reference policy; should grow slowly
6. `train/grad_norm` — non-zero but not exploding
7. `train/entropy` — gradual decrease is fine; crashing to 0 fast is exploration collapse
8. `train/clip_ratio/region_mean` — fraction of tokens PPO-clipped; healthy ~0–10%

## Resilience

`training/env_adapter/client.py` retries transient `httpx.TransportError`, `TimeoutException`, and 5xx responses (500, 502, 503, 504) on `/step`, `/reset`, `/state` with backoff `1s, 3s, 8s, 20s` (~32s budget). After exhaustion it raises `EnvUnavailableError`. The trainer catches that, emits a zero-reward dead step with `info[step_error]`, ends the episode for that prompt with `done=True`, and the rest of the batch continues. 4xx is not retried (caller bug, not transient).

## Publishing

Set on the job:

```bash
--env "PUBLISH_ON_FINISH=true"
--env "PUBLISH_MERGED=true"
```

Targets are read from `training/config/publish.yaml`:

```yaml
adapter_repo_id: your-namespace/faultline-red-qwen3-8b-lora
merged_repo_id:  your-namespace/faultline-red-qwen3-8b
private: false
license: mit
```

The merged push releases GPU memory before forking the merge subprocess to avoid OOM with vLLM still resident.

To push manually after a local run:

```python
from training.publish.push_adapter import push_adapter
from training.publish.push_merged import push_merged_model

push_adapter(folder_path="training/artifacts/checkpoints/checkpoint-60")
push_merged_model(adapter_path="training/artifacts/checkpoints/checkpoint-60")
```

## Local (single GPU)

For development, run the trainer directly against a local environment server:

```bash
pip install -e ".[training]"
python -m server.app &              # env on :8000
python training/jobs/train_entrypoint.py
```

Override the config:

```bash
TRAINING_CONFIG=training/config/training.smoke.yaml \
  python training/jobs/train_entrypoint.py
```

## Output Artifacts

- W&B run: prompts table + scalar metrics + LoRA adapter logged as a `model` artifact at every `save_steps` checkpoint and at end-of-training
- HF Hub: adapter and merged model in the repos from `publish.yaml`
- Local: `training/artifacts/checkpoints/checkpoint-N/` (gitignored)
