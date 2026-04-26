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

Our Round 1 submission was about debugging distributed systems. While working on that, one idea kept coming up: models trained for coding often seem to pick up useful security instincts along the way. That got us thinking, what if you built an environment specifically to train for that?


## What it is

Faultline is an adversarial environment for training agents on a live distributed system.


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

Training runs on HF Jobs (h200). This run completed and published both the LoRA adapter and a merged model to HF Hub.

One GRPO-specific note: loss is not the metric to watch. The useful signals are `reward/mean`, `reward_std`, and `completions/mean_length`. In this run, `reward_std` stayed above zero all the way through and `frac_reward_zero_std` finished at `0`, which means GRPO kept a real advantage signal instead of collapsing to identical generations.

## Results

Training finished at **60 steps**.

Final run metrics from W&B (`r4xtdrzj`):

| Metric | Final value |
|---|---|
| global step | 60 |
| reward mean | 1.2420 |
| reward std | 0.1926 |
| frac reward zero std | 0 |
| mean completion length | 2010.25 |
| KL | 0.0011709 |
| grad norm | 0.00607 |

From sampled W&B history, the best observed reward in the run was **1.4002** at step **46**. The first sampled non-null reward point was **1.1327** at step **3**, and it finished at **1.2420** at step **60**.

The important part is not the absolute number, it's that reward variance stayed alive for the whole run. `reward_std` finished at `0.1926` and `frac_reward_zero_std` finished at `0`, so GRPO never lost the signal it needed to distinguish better attacks from worse ones.

Both publish targets completed successfully:

- Adapter: `Veer15/wargames-red-qwen3-8b-lora`
- Merged model: `Veer15/wargames-red-qwen3-8b`

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
- W&B training run: https://api.wandb.ai/links/viraj-shah1503-none/l5wy9mu5
- Trained adapter: https://huggingface.co/Veer15/wargames-red-qwen3-8b-lora
- Merged model: https://huggingface.co/Veer15/wargames-red-qwen3-8b
