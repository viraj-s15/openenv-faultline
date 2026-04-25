"""Generate a HuggingFace Hub README for the trained model artifacts."""

from __future__ import annotations


def build_model_card_text(
    repo_id: str,
    base_model: str,
    artifact_kind: str,
    license: str = "mit",
    tags: tuple[str, ...] = ("grpo", "wargames", "red-team", "trl"),
) -> str:
    tag_block = "\n".join(f"- {tag}" for tag in tags)
    return (
        "---\n"
        f"base_model: {base_model}\n"
        f"license: {license}\n"
        "library_name: peft\n"
        "tags:\n"
        f"{tag_block}\n"
        "---\n\n"
        f"# {repo_id}\n\n"
        f"Artifact kind: {artifact_kind}\n\n"
        f"Base model: `{base_model}`\n\n"
        "Trained on the WarGames GRPO curriculum against the scripted Blue defender.\n"
        "Red Team agent that issues bash commands as JSON `{\"command\": \"...\"}`.\n"
    )
