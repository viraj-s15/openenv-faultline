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
| `model.base_model` | `Qwen/Qwen3.5-9B` | Base model to fine-tune |
| `model.lora_rank` | 16 | LoRA rank |
| `model.load_in_4bit` | true | QLoRA for memory efficiency |
| `trainer.learning_rate` | 5e-6 | GRPO learning rate |
| `trainer.num_generations` | 4 | Samples per prompt |
| `trainer.max_completion_length` | 128 | Max tokens per completion |
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

Set `wandb.enabled: false` to disable. When enabled, TRL reports loss, reward, and rollout stats to the wandb dashboard automatically.

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
  --secrets HF_TOKEN \
  python training/jobs/train_entrypoint.py
```

Set `TRAINING_CONFIG` env var to override the default config path:

```bash
export TRAINING_CONFIG=training/config/training.base.yaml
```

### How it works

1. Loads base model with unsloth + LoRA adapters
2. Selects curriculum tasks based on current trainer step
3. For each batch, runs rollouts against the live environment (Red agent sends commands, gets rewards)
4. GRPO updates model weights using rollout rewards
5. Checkpoints saved to `training/artifacts/checkpoints/`

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

push_adapter(
    repo_id="your-org/faultline-red-lora",
    folder_path="training/artifacts/checkpoints/checkpoint-XXX",
    private=False,
)
```

### Push merged model

```python
from training.publish.push_merged import export_merged_model
from training.publish.model_card import build_model_card_text

output_dir = export_merged_model(
    base_model="Qwen/Qwen3.5-9B",
    adapter_path="training/artifacts/checkpoints/checkpoint-XXX",
    output_dir="training/artifacts/merged",
)
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
| `WANDB_MODE` | `disabled` | Set to `online` to log to wandb |
| `WANDB_PROJECT` | `faultline` | wandb project name |
| `WANDB_ENTITY` | your default | wandb entity (team/user) |
| `WANDB_NAME` | auto | Run name (auto-generated if not set) |