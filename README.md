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

Constant bugs and security regressions in distributed systems were one of our most persistent internal challenges at Fibr AI — that's what we tackled in Round 1. During that work, a passing observation stuck with us: Dario Amodei (Anthropic) once noted in a podcast that Claude had been trained for coding but had developed an unusually strong grasp of security. It wasn't designed in. It emerged.

That got us thinking. If a model can stumble into security intuition, what would happen if you *deliberately engineered an environment to maximize it*?

We tried to build that here. We call it **Faultline**.

---

## What It Is

The same environment, two valid roles: an agent that learns to attack, and an agent that learns to defend. The environment doesn't care which side you train — it measures damage and recovery in the same units. For this submission we focus on the **attacker**, trained end-to-end via GRPO.

Two roles the environment supports:

- **The Attacker (Red)** — probes for vulnerabilities relentlessly. Queue poisoning, config corruption, service cascade, lock starvation. Raw bash. No guardrails.
- **The Defender (Blue)** — responds. Restarts services, restores configs, clears poison, monitors metrics. Escalating curriculum from passive to reactive.

The mesh the agents operate on:

- **Gateway** (`:3000`) — HTTP orchestration layer. Routes requests, tracks success rate and p99 latency. The primary health signal.
- **Auth** (`:3001`) — Authentication service with configurable delay. A latency injection target.
- **Redis** (`:6379`) — Job queue and distributed lock store. Poisonable, floodable, starve-able.
- **Worker** — Async job consumer. Reads from the Redis queue, writes to SQLite. Kill it and the queue backs up; corrupt the queue and it stalls silently.
- **SQLite** — Persistence sink. Slow writes propagate back up the chain.

Every component is live inside the container — real processes, real sockets, real failure modes. When the worker dies the queue depth climbs. When Auth slows the gateway p99 spikes. Attacks have to understand the system to do real damage.

---

## The Environment

### What the agent sees

Each step the Red agent receives a structured observation:

- `command_output` — merged stdout/stderr from its last bash command
- Live system metrics: gateway success rate, p99 latency, queue depth, worker restarts
- Episode state: step count, task name, Blue level, done flag

### What the agent does

One action per step: a raw bash command string.

```json
{"command": "redis-cli LPUSH job_queue '{broken'"}
```

The environment executes it with `subprocess.run(command, shell=True)` inside the isolated container. The agent can do anything a shell user can: read logs, inspect Redis, corrupt configs, kill processes, cover tracks.

### What the agent is rewarded for

Dense per-step reward in `[0, 1]` based on how much damage the action caused *and how well it survived Blue's defensive tick*:

| Component | What it measures |
|---|---|
| `success_rate_drop` | Gateway success rate fell |
| `latency_spike` | p99 latency increased |
| `queue_backup` | Redis job queue depth grew |
| `stealth_bonus` | Damage persisted after Blue's response |
| `no_op_penalty` | Metrics didn't move |
| `repeat_penalty` | Repeated a recent command |

The stealth bonus is the interesting one: it's not enough to break things, you have to break them in a way that survives the defender's next move.

### The Blue curriculum

During training, Red fights an escalating scripted defender — no LLM inference needed, so no cost and no instability:

| Level | Behavior |
|---|---|
| L0 | No defense. Red attacks a naked system. |
| L1 | Auto-restart crashed services every 5s. |
| L2 | L1 + config watchdog (detects and restores modified configs). |
| L3 | L2 + queue sanitizer + stale lock cleaner. |
| L4 | Event-triggered rollback when metrics cross thresholds. |

L4 is the intended training target. To score well at L4, Red must execute coordinated multi-vector attacks that overwhelm the rule-based logic before it can respond.

The final evaluation mode replaces the scripted Blue with a **prompted Blue LLM** — an incident commander agent with defensive bash access. That's the live-fire showdown.

### The mesh

```
gateway:3000 ──▶ auth:3001
     │
     ▼
redis:6379 ──▶ worker ──▶ sqlite
```

Gateway, Auth, Redis, Worker, SQLite — all running live inside the container. `metrics_poller` tracks health continuously. Blue acts once per Red step.

---

## Training

Red is trained with **GRPO** (Group Relative Policy Optimization) on a Qwen3-8B base, using LoRA for efficiency. The training loop connects directly to the live environment — no static dataset, no precomputed rewards.

Key hyperparameters: `num_generations=4` (group size for advantage estimation), `max_steps_per_episode=4` (bash commands per rollout), `lora_rank=16`.

Training runs on HF Jobs (h200). The full pipeline publishes the LoRA adapter and a merged model to HF Hub on completion.

### Metrics that matter

GRPO is a policy-gradient method. Loss is not a progress metric — it's a surrogate that doesn't decrease monotonically with improvement. The signals worth watching:

- **`reward/mean`** — trending up is the primary signal
- **`reward_std`** — must stay above zero; if it collapses, GRPO has no advantages to learn from
- **`frac_reward_zero_std`** — fraction of groups where all generations scored identically; should stay near zero
- **`completions/mean_length`** — sudden collapse means the model learned to emit nothing
- **`kl`** — slow divergence from reference is healthy; sudden jump is not

### Resilience

The env client retries transient HTTP 5xx and transport errors with backoff `(1s, 3s, 8s, 20s)`. On permanent failure, the episode ends early with a zero reward for that prompt — the rest of the batch continues. 4xx is not retried.

---

## Results

*[W&B run `r4xtdrzj` — training in progress. Reward curves, before/after transcripts, and Red vs Blue LLM showdown logs will be added on completion.]*

**Early signal (step 4/60):**

| Metric | Step 1 | Step 4 |
|---|---|---|
| reward mean | 1.09 | 1.32 |
| reward std | 0.036 | 0.48 |
| env reward mean | 0.34 | 0.57 |
| step errors | 0 | 0 |

reward_std moving from near-zero to 0.48 in four steps means the model is already generating meaningfully differentiated outputs — GRPO has a useful advantage signal to work with.

---

## Why It Matters

Security intuition in LLMs is mostly accidental today. Models pick it up as a side effect of training on code and CVEs, but no one has deliberately engineered an environment to *cultivate* it.

Faultline is an attempt at that. A place where an agent is forced to develop real attack intuition — not trivia recall, but the kind of systematic probing that finds the interaction between a misconfigured timeout and a Redis queue depth that a human pentester might catch after three hours of poking around.

If that works at 8B parameters in 60 training steps, it suggests the capability gap is mostly about environment design, not model scale. That's an interesting thing to know.

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
