## Quick deploy via HF CLI

Use the **merged model** repo for inference endpoints, not the LoRA adapter repo:
- `Veer15/wargames-red-qwen3-8b`
- not `Veer15/wargames-red-qwen3-8b-lora`

1. Log in and inspect the endpoint catalog options:

```bash
hf auth login
hf endpoints catalog ls | head -50
```

2. Create the endpoint. **Use `nvidia-l40s x4` minimum** — see Gotchas below for why smaller shapes fail:

```bash
hf endpoints deploy faultline-red \
  --repo Veer15/wargames-red-qwen3-8b \
  --task text-generation \
  --framework pytorch \
  --accelerator gpu \
  --instance-size x4 \
  --instance-type nvidia-l40s \
  --vendor aws \
  --region us-east-1 \
  --min-replica 0 \
  --max-replica 1 \
  --scale-to-zero-timeout 15
```

3. Switch the image to TGI so the endpoint exposes OpenAI-compatible `/v1/chat/completions`. The default `transformers` pipeline container does **not** route that path. The CLI cannot set the image, so PUT the endpoint config directly:

```bash
python - <<'PY'
import os, requests
token = os.environ['HF_TOKEN']
url = 'https://api.endpoints.huggingface.cloud/v2/endpoint/Veer15/faultline-red'
headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
current = requests.get(url, headers=headers).json()
body = {
    'compute': current['compute'],
    'model': {**current['model'], 'image': {'tgi': {'url': 'ghcr.io/huggingface/text-generation-inference:latest'}}},
    'type': current['type'],
}
r = requests.put(url, headers=headers, json=body)
print(r.status_code, r.text[:400])
PY
```

4. Wait for it to come up and copy the URL:

```bash
hf endpoints describe faultline-red
```

5. Run the benchmark against that endpoint with the merged model id:

```bash
docker run --rm \
  -e HF_TOKEN \
  -e API_BASE_URL="<YOUR_ENDPOINT_URL>/v1" \
  -e BLUE_API_BASE_URL="<YOUR_ENDPOINT_URL>/v1" \
  -v "$PWD/outputs:/home/user/app/outputs" \
  -v "$PWD/evals:/home/user/app/evals:ro" \
  wargames-inference \
  python evals/run_red_blue_benchmark.py \
    --provider hf \
    --models "Veer15/wargames-red-qwen3-8b" \
    --max-steps 30
```

6. Pause or delete it when done (l40s x4 is not cheap):

```bash
hf endpoints pause faultline-red
# or
hf endpoints delete faultline-red
```

## Notes

- The HF router path (`https://router.huggingface.co/v1`) only works if the model is available through an enabled provider. A dedicated Inference Endpoint avoids that dependency.
- The merged repo metadata was patched for chat/text-generation compatibility. Use the merged repo for benchmarking.

## Gotchas hit during deploy

- **Host RAM 30 GiB cap on small instances.** `nvidia-l4 x1` only allows 30 GiB host RAM. The default container loads weights on CPU first, and Qwen3-8B in bf16 doesn't fit during load. Failure surfaces as `Memory limit exceeded (30.0G)`. Bump instance size up — `x4` clears it.
- **GPU memory on a single L4 is too small.** L4 = 22 GiB GPU. After fixing host RAM, the next failure was `torch.OutOfMemoryError: CUDA out of memory` while moving weights to device. Use `nvidia-l40s` (48 GiB) or larger.
- **Tokenizer config field shape.** The published `tokenizer_config.json` had `extra_special_tokens` as a list. The HF inference container's `transformers` version expects a dict and crashes with `AttributeError: 'list' object has no attribute 'keys'` in `_set_model_specific_special_tokens`. Fix: rename the field to `additional_special_tokens` on the merged repo to match the base model. The base `Qwen/Qwen3-8B` uses `additional_special_tokens` and has no `extra_special_tokens` field — match that shape.
- **Default container has no OpenAI chat route.** The HF default `huggingface` pipeline image returns `Not Found` for `/v1/chat/completions`. Switch `model.image` to `tgi` (`ghcr.io/huggingface/text-generation-inference:latest`) via PUT on `/v2/endpoint/<ns>/<name>`. CLI flags don't expose this.
- **Control plane flakiness.** `hf endpoints update` 500s while the endpoint is in `failed` state. Workaround: `hf endpoints delete <name> --yes`, wait ~90s, then `hf endpoints deploy` fresh.
- **`/whoami-v2` rate limit.** Rapid `hf endpoints describe` calls trip a strict rate limit. Poll at ~75–90s intervals.
- **CLI scale-to-zero requirement.** `--scale-to-zero-timeout` is only accepted with `--min-replica 0`.
- **CLI framework values.** `--framework` accepts `custom`, `pytorch`, `llamacpp` only. `vllm` and `tgi` are not valid here; switch to TGI via the image PUT above.

## Final working config

- name: `faultline-red`
- vendor / region: `aws / us-east-1`
- accelerator: `gpu`
- instance: `nvidia-l40s x4`
- framework: `pytorch`
- task: `text-generation`
- image: `ghcr.io/huggingface/text-generation-inference:latest` (TGI)
- min / max replicas: `0 / 1`
- scale-to-zero timeout: `15 min`