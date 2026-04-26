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

Distributed systems breaking under attack was one of Fibr AI's most persistent internal headaches — that's what we tackled in Round 1. While building that, something Dario Amodei (Anthropic) said in a podcast kept nagging at us: Mythos was trained for coding, but somewhere along the way it picked up an unusually strong intuition for security. Nobody planned that. It just emerged.

That got us thinking. If a model can stumble into security intuition, what happens if you build an environment specifically designed to force it?

That's Faultline.

---

## The idea

One environment, two roles. The same mesh, the same reward signal, the same bash interface — but you can put an attacker on one side or a defender on the other. Damage and recovery are measured in the same units, so the environment doesn't care which side you're training.

For this submission we train the attacker.

Red gets raw bash access to a live distributed service mesh and has to learn to actually break it — not guess at flags, not navigate a grid, but figure out that poisoning the Redis job queue with malformed JSON will stall the worker, which backs up the queue, which eventually takes down the gateway. That chain of causation has to be discovered. The model has to earn the reward.

During training, Blue is a rules-based curriculum — five escalating difficulty levels that go from doing nothing (L0) to event-triggered rollbacks the moment a metric drops (L4). This keeps training cost predictable and avoids the instability of training two models simultaneously. At inference, the scripted Blue is swapped out for a prompted LLM incident commander — that's the live-fire evaluation.

---

## The mesh

Everything runs live inside the container:

- **Gateway** (`:3000`) — the HTTP orchestration layer. Tracks success rate and p99 latency. If this goes red, the episode is over.
- **Auth** (`:3001`) — authentication with configurable delay. Slow it down and the gateway p99 starts climbing before anyone notices.
- **Redis** (`:6379`) — job queue and distributed lock store. Push broken JSON into the queue and the worker stalls silently. Flood it and locks pile up.
- **Worker** — reads from the Redis queue, writes to SQLite. Kill it and the queue depth starts climbing. Corrupt what it reads and it dies on its own after restart.
- **SQLite** — the persistence sink. Slow writes propagate back up.

When the worker dies the queue backs up. When Auth slows the gateway p99 spikes. The reward function sees all of it.

---

## What the agent does

One action per step: a bash command.

```json
{"command": "redis-cli LPUSH job_queue '{broken'"}
```

The environment runs it with `subprocess.run(command, shell=True)`. Read logs, inspect Redis, corrupt a config, kill a process, cover tracks — anything a shell user can do. The observation comes back with the command output, current metrics, and episode state.

Reward is dense per-step, in `[0, 1]`:

| Component | What it measures |
|---|---|
| `success_rate_drop` | Gateway success rate fell |
| `latency_spike` | p99 latency increased |
| `queue_backup` | Redis job queue depth grew |
| `stealth_bonus` | Damage persisted after Blue's response |
| `no_op_penalty` | Metrics didn't move |
| `repeat_penalty` | Repeated a recent command |

The stealth bonus is the part that forces real learning. Breaking things is easy. Breaking them in a way that survives the defender's next move requires understanding the system.

---

## Training

Red is trained with GRPO on Qwen3-8B, LoRA rank 16. The training loop connects directly to the live environment — no static dataset, no precomputed rewards. Four bash commands per rollout, four generations per group for advantage estimation.

During training Blue is the scripted curriculum:

| Level | What it does |
|---|---|
| L0 | Nothing. The system is naked. |
| L1 | Restarts crashed services every 5 seconds. |
| L2 | L1 plus a config watchdog that detects and restores modified configs. |
| L3 | L2 plus queue sanitizer and stale lock cleaner. |
| L4 | Event-triggered rollback the moment a metric crosses a threshold. |

L4 is the target. To get reward at L4 the model has to coordinate multiple attack vectors faster than the rule-based defender can respond — spike latency while simultaneously taking down Auth, for example, so the rollback triggers on the wrong root cause.

Training runs on HF Jobs (h200). The full pipeline auto-publishes the LoRA adapter and a merged model to HF Hub on completion.

One thing worth knowing about GRPO: loss is not a useful progress metric. It's a policy-gradient surrogate and it doesn't decrease as the model improves. Watch `reward/mean`, `reward_std` (if this collapses to zero GRPO has nothing to learn from), and `completions/mean_length` (sudden drop means the model learned to say nothing).

---

## Results

*W&B run `r4xtdrzj` is in progress. Full reward curves, before/after transcripts, and the Red vs Blue LLM showdown log will be here on completion.*

Early signal at step 4/60:

| Metric | Step 1 | Step 4 |
|---|---|---|
| reward mean | 1.09 | 1.32 |
| reward std | 0.036 | 0.48 |
| env reward mean | 0.34 | 0.57 |
| step errors | 0 | 0 |

`reward_std` going from near-zero to 0.48 in four steps means the model is already generating meaningfully different outputs — GRPO has a real signal to work with.

---

## Why it matters

Security intuition in LLMs right now is almost entirely accidental. Models pick it up from code and CVE writeups in pretraining, but nobody has built an environment specifically to develop it through interaction.

Faultline is a bet that the capability gap here is mostly about environment design, not model scale. If a fine-tuned 8B model can learn to find the interaction between a misconfigured auth delay and a Redis lock timeout in 60 training steps on a $13 compute budget, that's worth knowing.

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
