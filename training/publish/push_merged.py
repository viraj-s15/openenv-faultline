from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def export_merged_model(base_model: str, adapter_path: str, output_dir: str) -> str:
    model = AutoModelForCausalLM.from_pretrained(base_model)
    peft_model = PeftModel.from_pretrained(model, adapter_path)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(output_dir)
    return output_dir
