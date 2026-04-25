"""Load the base model with unsloth + GRPO-patched LoRA adapter."""

from __future__ import annotations


def load_training_model(settings: dict):
    from unsloth import FastLanguageModel, PatchFastRL

    PatchFastRL(algorithm="grpo", FastLanguageModel=FastLanguageModel)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=settings["model"]["base_model"],
        max_seq_length=settings["model"]["max_seq_length"],
        load_in_4bit=settings["model"]["load_in_4bit"],
        fast_inference=settings["trainer"]["use_vllm"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=settings["model"]["lora_rank"],
        target_modules=settings["model"]["target_modules"],
        lora_alpha=settings["model"]["lora_alpha"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=settings["model"].get("random_state", 3407),
        max_seq_length=settings["model"]["max_seq_length"],
    )
    return model, tokenizer
