---
title: wargames-openenv
sdk: docker
app_port: 8000
colorFrom: red
colorTo: indigo
short_description: WarGames red-vs-blue distributed systems env
---

# WarGames

WarGames is an OpenEnv environment for teaching agents to interact with a live distributed system through bash commands.

Phase 0 ports the Round 1 service mesh into a root-deployable project. Phase 1 gives the Red agent raw bash access through the `command` action field. Phase 2 adds scripted Blue training rules and one prompted Blue LLM showdown mode.

- Gateway on port `3000`
- Auth on port `3001`
- Redis on port `6379`
- OpenEnv API on port `8000`

The Python package lives under `src/wargames_env`, and the mesh services live under `mesh`.

## Local Run

```bash
APP_ROOT="$PWD" MESH_ROOT="$PWD/mesh" ./start.sh
```

Then call the environment:

```bash
curl -X POST http://localhost:8000/reset
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"command": "curl -sf localhost:3000/health"}'
```

In Docker, `/mesh` points at the app-local mesh directory. On local machines, `start.sh` attempts to create the same `/mesh` link when permissions allow it. If that link is unavailable, use the exported `MESH_ROOT` path inside Red commands, for example `cat "$MESH_ROOT/gateway/config.json"`.

## Docker And Spaces

The root `Dockerfile` is the Hugging Face Spaces hosting image. It starts Redis, the Bun mesh services, and the FastAPI OpenEnv server through `start.sh`.

```bash
make build-space
make run-space
```

In another shell, smoke test the hosted environment:

```bash
make smoke-space
```

Hugging Face Spaces uses the `sdk: docker` and `app_port: 8000` metadata above. Keep that port aligned with `openenv.yaml`, `Dockerfile`, and the default `PORT` in `start.sh`.

`Dockerfile.inference` preserves the local Dockerized benchmark/evaluation image, including `inference.py`:

```bash
make build-inference
```

LLM provider secrets are only required for LLM-driven tasks such as `phase-2-blue-llm-showdown`; basic `/health`, `/reset`, and `/step` calls do not require them.

## Red Action Schema

The Red agent sends a single raw bash command:

```json
{"command": "redis-cli KEYS '*'"}
```

`/step` executes the command with `subprocess.run(command, shell=True)`. The response includes merged stdout/stderr in `observation.command_output`, current metrics, process status, and command metadata in `info`:

- `exit_code`
- `timed_out`
- `command`
- `duration_ms`
- `blue_actions`
- `reward`
- `termination_reason`
- `error`

## Phase 4 Environment Contract

`WarGamesEnv` is the canonical OpenEnv environment. `reset(task_name=...)` applies task config, including the server-side `max_steps`, and `/state` returns the authoritative episode state:

```json
{
  "episode_id": "episode-1",
  "task_name": "phase-2-blue-l4",
  "step_count": 1,
  "max_steps": 10,
  "blue_mode": "scripted",
  "blue_level": 4,
  "metrics": {},
  "process_status": {}
}
```

`/step` sets `done` when either:

- `termination_reason=max_steps`: the task step budget is exhausted.
- `termination_reason=mesh_down`: critical services `gateway`, `auth`, and `worker` are all stopped.

When the mesh is down, `info.error` contains a compact failure string for inference logs.

## Phase 3 Red Reward

`/step` now returns dense per-step Red reward in `[0.0, 1.0]`. The environment snapshots metrics before Red, after Red, and after Blue so the score can distinguish immediate damage from damage that persists through defense.

Reward components:

- `success_rate_drop`: gateway success rate decreased.
- `latency_spike`: gateway p99 latency increased.
- `queue_backup`: Redis `job_queue` depth increased.
- `stealth_bonus`: damage persisted after Blue's tick.
- `no_op_penalty`: metrics did not materially change.
- `repeat_penalty`: Red repeated a recent command.

`info.reward` includes the total, component values, weighted component values, and metric snapshots:

```json
{
  "reward": {
    "total": 0.4,
    "components": {"success_rate_drop": 0.5},
    "metrics_before": {},
    "metrics_after_red": {},
    "metrics_after_blue": {}
  }
}
```

## Phase 2 Blue Defense

Scripted Blue levels are only for Red training curriculum:

- `phase-2-blue-l0`: no Blue actions.
- `phase-2-blue-l1`: restart stopped mesh services.
- `phase-2-blue-l2`: restore baseline mesh configs and send SIGHUP.
- `phase-2-blue-l3`: sanitize malformed Redis queue entries and clear stale worker locks.
- `phase-2-blue-l4`: trigger rollback metadata when success rate, latency, or queue depth crosses thresholds.

The Blue LLM has one evaluation mode:

- `phase-2-blue-llm-showdown`: one prompted incident commander tick runs after each Red action.

Use a curriculum task by passing `task_name` to reset:

```bash
curl -X POST "http://localhost:8000/reset?task_name=phase-2-blue-l4"
curl -X POST "http://localhost:8000/reset?task_name=phase-2-blue-llm-showdown"
```

The Blue LLM provider uses OpenAI-compatible environment variables. Defaults match the Round 1 inference semantics:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="..."
```

For OpenRouter, use the same code path:

```bash
export API_BASE_URL="https://openrouter.ai/api/v1"
export MODEL_NAME="qwen/qwen-2.5-72b-instruct"
export API_KEY="..."
```

Blue-specific overrides are available with `BLUE_API_BASE_URL`, `BLUE_MODEL_NAME`, `BLUE_API_KEY`, `BLUE_TEMPERATURE`, and `BLUE_MAX_COMPLETION_TOKENS`.

Run Red inference with:

```bash
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" HF_TOKEN="..." python inference.py
```

Set `TASKS_CSV` to choose tasks, for example:

```bash
TASKS_CSV="phase-2-blue-l4,phase-2-blue-llm-showdown" python inference.py
```

`iptables` and `systemctl` are not assumed to work in the container runtime. Blue uses mesh-native actions by default: process restarts, config restore plus SIGHUP, Redis cleanup, log inspection, and metrics-triggered rollback.

## Phase 1 Example Commands

Recon:

```bash
cat /mesh/gateway/config.json
redis-cli KEYS '*'
tail -20 /tmp/worker.log
curl localhost:3000/health
```

Attack:

```bash
redis-cli LPUSH job_queue '{broken'
echo '{"delay_ms": 1500}' > /mesh/auth/config.json
kill -9 $(pgrep worker)
```

Stealth:

```bash
truncate -s 0 /tmp/worker.log
```

Phase 1 intentionally allows destructive commands inside the isolated environment. Phase 2 and Phase 3 add Blue defense and dense Red reward scoring around those actions.

## Training

GRPO training, Hugging Face Job launch steps, adapter publishing, merged-model export, and Space deployment assets live under `training/`.

Start with `training/README.md`.
