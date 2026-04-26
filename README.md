---
title: wargames-openenv
sdk: docker
app_port: 8000
colorFrom: red
colorTo: indigo
short_description: Faultline — adversarial Red vs Blue on a live mesh
---

# Faultline

> *"The only winning move is to learn."*

Distributed systems breaking under attack was one of Fibr AI's most persistent internal headaches. That's what we tackled in Round 1. During that work, one idea kept coming up: models trained for coding often seem to pick up useful security instincts along the way. That got us thinking — what if you built an environment specifically to train for that?


## What it is

Faultline is an adversarial environment for training agents on a live distributed system.

Tagline: *An 8B model walks into a server room...*

The same environment supports two roles:
- **Red** attacks the system.
- **Blue** defends it.

For this hackathon, we train **Red**.

During training, Blue is not another model. It's a rules-based curriculum with five difficulty levels. That keeps the training loop stable and cheap. During inference, Blue can be another agent. That's the actual Red-vs-Blue setup.

## The system

The mesh is small, but it behaves like a real distributed system:

- **Gateway** (`:3000`) handles HTTP traffic. We track success rate and p99 latency here.
- **Auth** (`:3001`) handles authentication and has configurable delay.
- **Redis** (`:6379`) is the job queue and distributed lock store.
- **Worker** consumes jobs from Redis and writes results to SQLite.
- **SQLite** is the persistence layer.

Those pieces interact in ways the model has to learn. Slow Auth and gateway latency spikes. Kill the Worker and queue depth grows. Poison Redis and the Worker stalls.

## What the agent can do

One step is one bash command.

```json
{"command": "redis-cli LPUSH job_queue '{broken'"}
```

The environment executes that command with `subprocess.run(command, shell=True)` inside the container. The agent can inspect logs, read configs, poison Redis, kill processes, or clean up its traces.

The observation includes:
- command output
- current system metrics
- episode state

Reward is dense per step, in `[0, 1]`. Or, less formally: *how much damage did you cause?*

| Component | What it measures |
|---|---|
| `success_rate_drop` | Gateway success rate fell |
| `latency_spike` | p99 latency increased |
| `queue_backup` | Redis job queue depth grew |
| `stealth_bonus` | Damage persisted after Blue's response |
| `no_op_penalty` | Metrics didn't move |
| `repeat_penalty` | Repeated a recent command |

The important part is `stealth_bonus`. It's not enough to break the system. The damage has to survive the defender's response.

## Training

We train Red with GRPO on Qwen3-8B using LoRA rank 16. The loop talks directly to the live environment. No static dataset. No precomputed rewards.

Blue during training is the scripted curriculum:

| Level | What it does |
|---|---|
| L0 | Nothing. |
| L1 | Restarts crashed services every 5 seconds. |
| L2 | L1 + restores modified configs. |
| L3 | L2 + sanitizes the queue and clears stale locks. |
| L4 | Rolls back as soon as metrics cross thresholds. Final boss: **YOU SHALL NOT PASS**. |

L4 is the target.

Training runs on HF Jobs (h200). On completion, the job publishes both the LoRA adapter and a merged model to HF Hub.

One GRPO-specific note: loss is not the metric to watch. The useful signals are `reward/mean`, `reward_std`, and `completions/mean_length`. If `reward_std` collapses to zero, GRPO has no advantage signal to learn from.

## Results

*W&B run `r4xtdrzj` is still running. Reward curves, before/after transcripts, and the Red-vs-Blue inference transcript go here once training finishes.*

Early signal at step 4/60:

| Metric | Step 1 | Step 4 |
|---|---|---|
| reward mean | 1.09 | 1.32 |
| reward std | 0.036 | 0.48 |
| env reward mean | 0.34 | 0.57 |
| step errors | 0 | 0 |

The main takeaway so far is that `reward_std` moved off zero quickly. The model is already producing meaningfully different actions, which means GRPO has signal to work with.

## Why this matters

Security intuition in current models is mostly accidental. Faultline is an attempt to train for it directly.

If an 8B model can learn useful attack behavior on a live service mesh in 60 steps on roughly a $13 budget, that says the missing piece may be environment design more than model scale.

---

## Try It

The environment runs on HF Spaces. Call it directly:

```bash
# Start an episode
curl -X POST https://veer15-wargames-env-train.hf.space/reset?task_name=phase-2-blue-l4

# Take a step
curl -X POST https://veer15-wargames-env-train.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"command": "redis-cli KEYS \"*\""}'
```

Or run locally:

```bash
APP_ROOT="$PWD" MESH_ROOT="$PWD/mesh" ./start.sh
```

Then hit `http://localhost:8000`.

**Evaluate a model against the environment:**

```bash
pip install -e ".[inference]"
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" HF_TOKEN="..." python inference.py
```

---

## Repository Layout

```
src/wargames_env/     ← OpenEnv Environment class, reward logic, Blue curriculum
mesh/                 ← Gateway, Auth, Worker (Bun/Node services)
training/             ← GRPO training pipeline, env adapter, publish
  config/             ← training.base.yaml, curriculum.l0-l4.yaml
  grpo/               ← TRL trainer wiring, reward functions, rollout
  env_adapter/        ← retry-resilient HTTP client to the env Space
  publish/            ← LoRA adapter + merged model push to HF Hub
  jobs/               ← HF Jobs bootstrap (run_on_hf.py, train_entrypoint.py)
docs/
  inference.md        ← running the benchmark / eval pipeline
  training.md         ← running a training job on HF
```

---

## Links

- HF Space (live environment): https://huggingface.co/spaces/Veer15/wargames-env-train
- W&B training run: https://wandb.ai/viraj-shah1503-none/faultline/runs/r4xtdrzj
- Trained adapter: `Veer15/wargames-red-qwen3-8b-lora` *(publishing on run completion)*
- Merged model: `Veer15/wargames-red-qwen3-8b` *(publishing on run completion)*
