from training.grpo.config import build_grpo_config
from training.grpo.reward_adapter import aggregate_episode_reward
from training.rollouts.episode_runner import run_episode


class LocalGenerationClient:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def build_prompt_dataset(tasks: list[str]) -> list[dict[str, str]]:
    return [{"task_name": task_name, "prompt": task_name} for task_name in tasks]


def reward_from_rollout(completions, trajectories, **kwargs):
    return [
        aggregate_episode_reward(trajectory.rewards, method="sum")
        for trajectory in trajectories
    ]


def make_rollout_func(llm_client, env_client, max_steps: int):
    def rollout_func(batch_prompts, **kwargs):
        trajectories = []
        for item in batch_prompts:
            task_name = item["task_name"] if isinstance(item, dict) else str(item)
            trajectories.append(
                run_episode(
                    llm_client=llm_client,
                    env_client=env_client,
                    task_name=task_name,
                    max_steps=max_steps,
                )
            )
        return trajectories

    return rollout_func


def build_trainer(model, tokenizer, dataset, reward_funcs, rollout_func, settings: dict):
    from trl import GRPOTrainer

    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        rollout_func=rollout_func,
        args=build_grpo_config(settings),
    )
