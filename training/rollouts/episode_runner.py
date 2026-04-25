from training.env_adapter.action_parser import parse_model_command
from training.env_adapter.observation_formatter import build_red_prompt
from training.rollouts.trajectory import EpisodeTrajectory, RolloutStep


def run_episode(llm_client, env_client, task_name: str, max_steps: int) -> EpisodeTrajectory:
    observation = env_client.reset(task_name)
    steps: list[RolloutStep] = []
    rewards: list[float] = []
    history: list[dict[str, object]] = []

    for step_num in range(1, max_steps + 1):
        prompt = build_red_prompt(observation, task_name, step_num, history)
        raw_completion = llm_client.generate(prompt)
        command = parse_model_command(raw_completion)
        result = env_client.step(command)
        steps.append(
            RolloutStep(
                step_num=step_num,
                prompt=prompt,
                raw_completion=raw_completion,
                command=command,
                reward=result.reward,
                done=result.done,
                info=result.info,
            )
        )
        rewards.append(result.reward)
        history.append(
            {
                "step": step_num,
                "command": command,
                "output": result.observation.command_output,
                "error": result.info.get("error"),
            }
        )
        observation = result.observation
        if result.done:
            break

    return EpisodeTrajectory(task_name=task_name, steps=steps, rewards=rewards)
