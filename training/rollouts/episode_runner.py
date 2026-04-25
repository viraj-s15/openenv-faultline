"""Run a full Red-vs-Blue episode against the live env."""

from __future__ import annotations

from training.env_adapter.action_parser import (
    ActionParseError,
    NO_COMMAND_PROVIDED,
    parse_model_command,
    parse_model_command_strict,
)
from training.env_adapter.observation_formatter import build_red_prompt
from training.rollouts.trajectory import EpisodeTrajectory, RolloutStep


def run_episode(
    llm_client,
    env_client,
    task_name: str,
    max_steps: int,
    strict_parse: bool = False,
) -> EpisodeTrajectory:
    """Run one episode end-to-end.

    `strict_parse=True` makes a malformed completion a no-op env step
    (`echo NO_COMMAND_PROVIDED`) but records the parse error in `info.parse_error`,
    so the reward function can see the failure mode without the env crashing.
    """
    observation = env_client.reset(task_name)
    steps: list[RolloutStep] = []
    rewards: list[float] = []
    history: list[dict[str, object]] = []

    for step_num in range(1, max_steps + 1):
        prompt = build_red_prompt(observation, task_name, step_num, history)
        raw_completion = llm_client.generate(prompt)
        parse_error: str | None = None
        if strict_parse:
            try:
                command = parse_model_command_strict(raw_completion)
            except ActionParseError as exc:
                parse_error = str(exc)
                command = NO_COMMAND_PROVIDED
        else:
            command = parse_model_command(raw_completion)

        result = env_client.step(command)
        info = dict(result.info or {})
        if parse_error is not None:
            info["parse_error"] = parse_error

        steps.append(
            RolloutStep(
                step_num=step_num,
                prompt=prompt,
                raw_completion=raw_completion,
                command=command,
                reward=result.reward,
                done=result.done,
                info=info,
            )
        )
        rewards.append(result.reward)
        history.append(
            {
                "step": step_num,
                "command": command,
                "output": result.observation.command_output,
                "error": info.get("error") or parse_error,
            }
        )
        observation = result.observation
        if result.done:
            break

    return EpisodeTrajectory(task_name=task_name, steps=steps, rewards=rewards)
