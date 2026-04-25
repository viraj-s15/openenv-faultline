# Running Red-vs-Blue Inference

This document covers Dockerized Red-vs-Blue benchmark runs for WarGames.

## Build the Inference Image

Build the local benchmark image from `Dockerfile.inference`:

```bash
make build-inference
```

This creates the `wargames-inference` image. Rebuild it after changing prompts, environment logic, reward logic, or benchmark code.

## Required Environment Variables

For OpenRouter runs, set:

```bash
OPENROUTER_API_KEY=...
```

For OpenCode Go runs, set:

```bash
OPENCODE_GO_KEY=...
OPENCODE_GO_URL=...
```

If the variables are stored in `.env`, load them into the shell without printing the file:

```bash
set -a
source .env
set +a
```

## Run a Single OpenRouter Benchmark

Use `evals/run_red_blue_benchmark.py` inside the inference image. The script creates a model-specific folder under `outputs/` and writes all result files there.

```bash
docker run --rm \
  -e OPENROUTER_API_KEY \
  -v "$PWD/outputs:/home/user/app/outputs" \
  -v "$PWD/evals:/home/user/app/evals:ro" \
  wargames-inference \
  python evals/run_red_blue_benchmark.py \
    --models "qwen/qwen3.5-9b" \
    --max-steps 30
```

The same model is used for Red and Blue in showdown mode.

## Run Multiple Models Sequentially

Pass multiple `--model` values. The harness runs them sequentially to reduce rate-limit pressure.

```bash
docker run --rm \
  -e OPENROUTER_API_KEY \
  -v "$PWD/outputs:/home/user/app/outputs" \
  -v "$PWD/evals:/home/user/app/evals:ro" \
  wargames-inference \
  python evals/run_red_blue_benchmark.py \
    --models "meta-llama/llama-3.1-8b-instruct,qwen/qwen3.5-9b" \
    --max-steps 30
```

## Run an OpenCode Go Benchmark

```bash
docker run --rm \
  -e OPENCODE_GO_KEY \
  -e OPENCODE_GO_URL \
  -v "$PWD/outputs:/home/user/app/outputs" \
  -v "$PWD/evals:/home/user/app/evals:ro" \
  wargames-inference \
  python evals/run_red_blue_benchmark.py \
    --models "minimax-m2.7" \
    --max-steps 30
```

## Output Files

Each run writes a timestamped folder like:

```text
outputs/docker_openrouter_qwen35_9b_YYYYMMDD_HHMMSS/
```

The important files are:

- `summary.csv`: one row per model with `actual_steps`, `final_score`, `max_reward`, `avg_reward`, and error state.
- `steps.csv`: one row per environment step with Red command, Red reasoning, reward, termination state, Blue actions, `services_affected`, and `services_restored`.
- `red_vs_blue.log`: full step-by-step terminal log, including Red commands, Blue commands, output, errors, and final status.

## Current Game Rules Reflected in Inference

The Red agent can use one direct process-kill command per episode. Later direct `kill`, `pkill`, or `killall` actions are rejected by the environment with `PROCESS_KILL_BUDGET_EXHAUSTED`.

In `phase-2-blue-llm-showdown`, Blue receives two repair turns after every Red action. Red reasoning is stored in benchmark outputs, but it is not shown to Blue.

The `services_affected` and `services_restored` fields are evaluation-only CSV columns. They are not added to either model prompt.
