def build_model_card_text(repo_id: str, base_model: str, artifact_kind: str) -> str:
    return (
        f"# {repo_id}\n\n"
        f"Artifact kind: {artifact_kind}\n\n"
        f"Base model: `{base_model}`\n\n"
        "Trained on the WarGames GRPO curriculum against the scripted Blue defender.\n"
    )
