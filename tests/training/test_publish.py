from pathlib import Path

from training.publish.model_card import build_model_card_text


def test_model_card_mentions_base_model_and_artifact_kind():
    text = build_model_card_text(
        repo_id="your-org/wargames-red-lora",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        artifact_kind="adapter",
    )

    assert "Qwen/Qwen2.5-7B-Instruct" in text
    assert "adapter" in text


def test_space_readme_template_declares_docker_sdk_and_app_port():
    text = Path("training/spaces/README.template.md").read_text()

    assert "sdk: docker" in text
    assert "app_port: 8000" in text
