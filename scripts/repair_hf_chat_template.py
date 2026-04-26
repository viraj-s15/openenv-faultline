#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

TARGET_REPOS = [
    "Veer15/wargames-red-qwen3-8b-lora",
    "Veer15/wargames-red-qwen3-8b",
]
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template.strip():
        raise SystemExit(f"Base model {BASE_MODEL} has no chat_template")

    api = HfApi(token=token)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for repo_id in TARGET_REPOS:
            src = hf_hub_download(repo_id=repo_id, filename="tokenizer_config.json", repo_type="model")
            data = json.loads(Path(src).read_text(encoding="utf-8"))
            data["chat_template"] = chat_template
            out = tmpdir / f"{repo_id.split('/')[-1]}-tokenizer_config.json"
            out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            api.upload_file(
                repo_id=repo_id,
                repo_type="model",
                path_in_repo="tokenizer_config.json",
                path_or_fileobj=str(out),
                commit_message="Add inline chat_template for HF chat compatibility",
            )
            print(f"patched {repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
