## Launch training on Hugging Face Jobs

```bash
hf jobs uv run \
  --flavor a100-large \
  --timeout 6h \
  --with "./[training]" \
  --secrets HF_TOKEN \
  python training/jobs/train_entrypoint.py
```

Set `TRAINING_CONFIG` if you want a non-default YAML file.
