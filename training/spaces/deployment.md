## Quick deploy via HF CLI

Use the **merged model** repo for inference endpoints, not the LoRA adapter repo:
- `Veer15/wargames-red-qwen3-8b`
- not `Veer15/wargames-red-qwen3-8b-lora`

1. Log in and inspect the endpoint catalog options:

```bash
hf auth login
hf endpoints catalog ls | head -50
```

2. Create the endpoint. Replace the hardware fields with a valid combination for your account/region:

```bash
hf endpoints deploy faultline-red \
  --repo Veer15/wargames-red-qwen3-8b \
  --task text-generation \
  --framework vllm \
  --accelerator gpu \
  --instance-size x1 \
  --instance-type nvidia-l4 \
  --vendor aws \
  --region us-east-1 \
  --scale-to-zero-timeout 15
```

3. Wait for it to come up and copy the URL:

```bash
hf endpoints describe faultline-red
```

4. Run the benchmark against that endpoint with the merged model id:

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

5. Pause or delete it when done:

```bash
hf endpoints pause faultline-red
# or
hf endpoints delete faultline-red
```

## Notes

- The HF router path (`https://router.huggingface.co/v1`) only works if the model is available through an enabled provider. A dedicated Inference Endpoint avoids that dependency.
- The merged repo metadata was patched for chat/text-generation compatibility. Use the merged repo for benchmarking.