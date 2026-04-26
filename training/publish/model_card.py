"""Generate a HuggingFace Hub README for the trained model artifacts."""

from __future__ import annotations


def build_model_card_text(
    repo_id: str,
    base_model: str,
    artifact_kind: str,
    license: str = "mit",
    tags: tuple[str, ...] = ("grpo", "faultline", "red-team", "trl"),
) -> str:
    tag_block = "\n".join(f"- {tag}" for tag in tags)
    library_name = "transformers" if artifact_kind == "merged" else "peft"
    pipeline_tag_block = "pipeline_tag: text-generation\n" if artifact_kind == "merged" else ""
    return (
        "---\n"
        f"base_model: {base_model}\n"
        f"license: {license}\n"
        f"library_name: {library_name}\n"
        f"{pipeline_tag_block}"
        "tags:\n"
        f"{tag_block}\n"
        "---\n\n"
        f"# {repo_id}\n\n"
        f"Artifact kind: {artifact_kind}\n\n"
        f"Base model: `{base_model}`\n\n"
        "Trained on the Faultline GRPO curriculum against the scripted Blue defender.\n"
        "Red Team agent that issues bash commands as JSON `{\"command\": \"...\"}`.\n"
    )
