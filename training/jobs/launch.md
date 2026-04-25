## Launch training on Hugging Face Jobs

```bash
hf jobs uv run \
  --flavor a100-large \
  --timeout 6h \
  --with "./[training]" \
  --secrets HF_TOKEN WANDB_API_KEY \
  python training/jobs/train_entrypoint.py
```

`WANDB_API_KEY` is required when `wandb.enabled: true` in the YAML config
(the entrypoint refuses to start without it). To run without W&B logging,
set `wandb.enabled: false` and drop the secret.

Set `TRAINING_CONFIG` if you want a non-default YAML file.
