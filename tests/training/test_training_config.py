from pathlib import Path

import yaml

from training.grpo.config import build_grpo_config


def test_training_base_config_declares_env_and_model_defaults():
    payload = yaml.safe_load(Path("training/config/training.base.yaml").read_text())

    assert payload["env"]["base_url"] == "http://localhost:8000"
    assert payload["model"]["base_model"] == "Qwen/Qwen3-8B"
    assert payload["trainer"]["algorithm"] == "grpo"


def test_publish_config_declares_adapter_and_merged_targets():
    payload = yaml.safe_load(Path("training/config/publish.yaml").read_text())

    assert payload["adapter_repo_id"].endswith("-lora")
    assert payload["merged_repo_id"] and "/" in payload["merged_repo_id"]


def test_space_config_declares_docker_sdk_and_app_port():
    payload = yaml.safe_load(Path("training/config/space.yaml").read_text())

    assert payload["sdk"] == "docker"
    assert payload["app_port"] == 8000


def test_build_grpo_config_maps_yaml_to_trl_settings():
    payload = {
        "trainer": {
            "output_dir": "training/artifacts/checkpoints",
            "learning_rate": 5e-6,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "num_generations": 4,
            "max_completion_length": 128,
            "temperature": 0.7,
            "beta": 0.001,
            "use_vllm": False,
        }
    }

    config = build_grpo_config(payload)

    assert config.output_dir == "training/artifacts/checkpoints"
    assert config.num_generations == 4
    assert config.max_completion_length == 128


def test_train_entrypoint_exists_and_references_training_base_config():
    text = Path("training/jobs/train_entrypoint.py").read_text()

    assert "training/config/training.base.yaml" in text
    assert "HF_TOKEN" in text
