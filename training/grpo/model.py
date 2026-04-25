"""Load the base model + tokenizer + LoRA config (no unsloth).

TRL's GRPOTrainer must own the PEFT wrapping so it can:
  - register a `ref` adapter for the reference model
  - sync LoRA deltas into the colocated vLLM weights via `vllm_generation.sync_weights()`
Pass the LoraConfig through `peft_config=`; do NOT pre-wrap with `get_peft_model`.
"""

from __future__ import annotations


def load_training_model(settings: dict):
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = settings["model"]
    base_model = model_cfg["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if model_cfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs.pop("torch_dtype", None)

    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=int(model_cfg["lora_rank"]),
        lora_alpha=int(model_cfg["lora_alpha"]),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(model_cfg["target_modules"]),
    )
    return model, tokenizer, lora_config
