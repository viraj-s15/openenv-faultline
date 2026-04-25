# Faultline — Training & Deployment

Train a Red Team agent using GRPO against a live distributed system, then deploy the result as a Hugging Face Space.

## Architecture

```
openenv.yaml              ← environment spec (tasks, reward range, client)
src/wargames_env/         ← FastAPI server + environment logic
src/wargames_env/server/  ← /reset, /step, /state, /health endpoints
training/
├── config/               ← training + curriculum + publish + space configs
├── env_adapter/           ← connects training loop to the env server
├── grpo/                  ← GRPO trainer, config, model loading, reward
├── prompts/               ← system prompt for the Red agent
├── rollouts/              ← episode runner, trajectory, sampler, transcript writer
├── jobs/                  ← HF Job entrypoints
├── publish/               ← push adapter/merged model to HF Hub
├── spaces/                ← Space deployment docs & templates
├── artifacts/             ← checkpoint output dir (gitignored)
└── notebooks/             ← Colab notebook for quick experiments
```

## Prerequisites

- Python ≥ 3.10
- A running WarGames environment server (the FastAPI app)
- Hugging Face account with write access to your target repos
- GPU: unsloth requires CUDA. A100 or better recommended.

## 1. Install

```bash
pip install -e ".[training]"
```

This pulls unsloth, trl, peft, transformers, torch, huggingface_hub, and wandb.

## 2. Start the Environment Server

The training loop talks to a live environment over HTTP. Start it first:

```bash
# From the repo root
python -m server.app
# or via the installed entrypoint:
server
```

The server listens on `http://localhost:8000` by default. Verify:

```bash
curl http://localhost:8000/health
```

## 3. Configure Training

### `training/config/training.base.yaml`

Controls model, LoRA, and trainer settings:

| Key | Default | What it does |
|---|---|---|
| `env.base_url` | `http://localhost:8000` | WarGames server address |
| `model.base_model` | `Qwen/Qwen3-8B` | Base model to fine-tune |
| `model.lora_rank` | 16 | LoRA rank |
| `model.load_in_4bit` | true | QLoRA for memory efficiency |
| `trainer.learning_rate` | 5e-6 | GRPO learning rate |
| `trainer.num_generations` | 4 | Samples per prompt |
| `trainer.max_completion_length` | 128 | Max tokens per completion |
| `trainer.use_vllm` | true | Required for env-grounded rollout_func (TRL >=0.25 only invokes rollout_func on the vLLM path) |
| `trainer.max_steps` | 1500 | Stops training at this trainer step |
| `trainer.save_steps` | 250 | Checkpoint cadence; the W&B artifact callback logs each checkpoint |
| `rollout.max_steps_per_episode` | 10 | Steps per rollout |
| `rollout.reward_aggregation` | sum | How to combine step rewards |

### `training/config/curriculum.l0-l4.yaml`

Defines which tasks the agent trains on at each trainer step:

```yaml
schedule:
  - until_step: 250
    tasks: [phase-2-blue-l0]    # sitting-duck defender
  - until_step: 500
    tasks: [phase-2-blue-l1]    # scripted restart defense
  - until_step: 750
    tasks: [phase-2-blue-l2]    # restart + config watchdog
  - until_step: 1000
    tasks: [phase-2-blue-l3]    # queue sanitizer + stale lock cleanup
  - until_step: 1500
    tasks: [phase-2-blue-l4]    # metric-triggered rollback
```

The agent faces progressively harder scripted Blue defenders. Override with:

```bash
export CURRICULUM_CONFIG=path/to/my-curriculum.yaml
```

### Weights & Biases

Training logs metrics to [wandb](https://wandb.ai) by default. Configure in `training/config/training.base.yaml`:

```yaml
wandb:
  enabled: true
  project: faultline
  entity: null        # uses your default wandb entity
  run_name: null      # auto-generated if null
```

Set `wandb.enabled: false` to disable. When enabled, TRL reports loss, reward, and rollout stats to the wandb dashboard automatically. The trainer also logs:

- per-step Red prompts and completions (`log_completions=true` is forced on when wandb is enabled, so the Red bash commands appear as a W&B Table)
- the LoRA adapter directory as a W&B `model` artifact at every `save_steps` checkpoint and at end-of-training (see `training/grpo/callbacks.py`)

When wandb is enabled the entrypoint requires `WANDB_API_KEY` to be set; pass it via `--secrets WANDB_API_KEY HF_TOKEN` on HF Jobs.

To log in before training:

```bash
wandb login
```

## 4. Train

### Local (single GPU)

```bash
python training/jobs/train_entrypoint.py
```

### Hugging Face Jobs (A100)

```bash
hf jobs uv run \
  --flavor a100-large \
  --timeout 6h \
  --with "./[training]" \
  --secrets HF_TOKEN WANDB_API_KEY \
  python training/jobs/train_entrypoint.py
```

Set `TRAINING_CONFIG` env var to override the default config path:

```bash
export TRAINING_CONFIG=training/config/training.base.yaml
```

### How it works

1. Loads base model with unsloth + LoRA adapters (vLLM-backed when `use_vllm: true`)
2. The curriculum callback hard-switches `trainer.train_dataset` when `state.global_step` crosses a `until_step` boundary in `curriculum.l0-l4.yaml`
3. For each batch, `rollout_func` runs one full env episode per generation; `reward_from_rollout` aggregates per-step env rewards (sum)
4. GRPO updates the LoRA weights using the group-relative advantages
5. Checkpoints saved to `training/artifacts/checkpoints/checkpoint-N`; each is logged as a W&B model artifact when wandb is enabled

## 5. Publish

### Config: `training/config/publish.yaml`

```yaml
adapter_repo_id: your-org/faultline-red-lora
merged_repo_id: your-org/faultline-red-merged
private: false
license: mit
```

### Push LoRA adapter

```python
from training.publish.push_adapter import push_adapter

# Picks adapter_repo_id, private, license from training/config/publish.yaml
# and base_model from training/config/training.base.yaml.
push_adapter(folder_path="training/artifacts/checkpoints/checkpoint-1500")
```

### Push merged model

```python
from training.publish.push_merged import push_merged_model

# Reads merged_repo_id/private/license from publish.yaml and base_model from
# training.base.yaml; merges in bf16 with low CPU mem usage and pushes the repo.
push_merged_model(adapter_path="training/artifacts/checkpoints/checkpoint-1500")
```

## 6. Deploy to Hugging Face Spaces

### Config: `training/config/space.yaml`

```yaml
title: Faultline Demo
emoji: "🛡️"
sdk: docker
app_port: 8000
space_repo_id: your-org/faultline-demo-space
artifact_mode: adapter   # or "merged"
```

### Required Space secrets

| Secret | Purpose |
|---|---|
| `HF_TOKEN` | HuggingFace API token |
| `MODEL_REPO_ID` | Repo with the trained adapter or merged model |
| `BASE_MODEL_NAME` | Base model name (required only in `adapter` mode) |

### Deployment steps

1. Create a new Docker Space on Hugging Face
2. Copy `training/spaces/README.template.md` as the Space README — update `sdk`, `app_port`, and metadata
3. Set secrets in Space Settings → Repository Secrets
4. The Space runs the WarGames FastAPI server on port 8000
5. Verify with `curl https://your-space.hf.space/health` and `/state`

The Space loads the trained model at startup (adapter or merged depending on `artifact_mode`) and serves the same `/reset`, `/step`, `/state` API that the training loop used. The model acts as the Blue Team defender.

## 7. Dashboard

A React dashboard for visualizing episode replays lives in `dashboard/`:

```bash
cd dashboard
npm install
npm run dev
```

On startup, the Vite plugin reads `outputs/` directories, converts CSV logs to JSON, and serves them at `/scenarios/`. Select a run from the index page to watch the Red vs Blue episode replay with service health, metrics, and reasoning.

## File Reference

| Path | Purpose |
|---|---|
| `training/config/training.base.yaml` | Model, LoRA, GRPO hyperparameters |
| `training/config/curriculum.l0-l4.yaml` | Task schedule by trainer step |
| `training/config/publish.yaml` | Target repos for adapter & merged model |
| `training/config/space.yaml` | HF Space deployment config |
| `training/env_adapter/client.py` | HTTP client talking to WarGames server |
| `training/env_adapter/observation_formatter.py` | Builds Red agent prompts from env state |
| `training/env_adapter/action_parser.py` | Parses model JSON output → bash command |
| `training/env_adapter/task_selector.py` | Selects tasks from curriculum schedule |
| `training/grpo/trainer.py` | GRPOTrainer wrapper, rollout func, reward adapter |
| `training/grpo/config.py` | Maps YAML config → TRL GRPOConfig |
| `training/grpo/model.py` | Loads base model with unsloth + LoRA |
| `training/grpo/reward_adapter.py` | Aggregates step rewards → episode reward |
| `training/rollouts/episode_runner.py` | Runs full episode: prompt → command → step → reward |
| `training/rollouts/trajectory.py` | Data classes for rollout steps and trajectories |
| `training/rollouts/sampler.py` | Batch rollout sampling |
| `training/rollouts/transcript_writer.py` | Writes episode logs to disk |
| `training/prompts/red_system_prompt.txt` | System prompt for Red Team agent |
| `training/jobs/train_entrypoint.py` | Main training script |
| `training/jobs/launch.md` | HF Jobs quickstart |
| `training/publish/push_adapter.py` | Push LoRA adapter to HF Hub |
| `training/publish/push_merged.py` | Merge adapter into base model, push to HF Hub |
| `training/publish/model_card.py` | Generate README for model repos |
| `training/spaces/` | Space deployment docs and templates |

## Environment Variables

| Variable | Default | Used by |
|---|---|---|
| `TRAINING_CONFIG` | `training/config/training.base.yaml` | train_entrypoint.py |
| `CURRICULUM_CONFIG` | `training/config/curriculum.l0-l4.yaml` | train_entrypoint.py |
| `HF_TOKEN` | required | HF Jobs, push_adapter, Space |
| `MODEL_REPO_ID` | — | Space (which model to load) |
| `BASE_MODEL_NAME` | — | Space (required in adapter mode) |
| `WANDB_MODE` | forced `online` when `wandb.enabled: true`; `disabled` otherwise | `training/grpo/config.py::configure_wandb` |
| `WANDB_API_KEY` | required when `wandb.enabled: true` | train_entrypoint.py |
| `WANDB_PROJECT` | `faultline` | wandb project name |
| `WANDB_ENTITY` | your default | wandb entity (team/user) |
| `WANDB_NAME` | auto | Run name (auto-generated if not set) |