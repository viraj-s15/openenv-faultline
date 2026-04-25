from training.rollouts.episode_runner import run_episode


def sample_grouped_episodes(llm_client, env_client, task_name: str, max_steps: int, group_size: int):
    return [
        run_episode(
            llm_client=llm_client,
            env_client=env_client,
            task_name=task_name,
            max_steps=max_steps,
        )
        for _ in range(group_size)
    ]
