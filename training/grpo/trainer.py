"""GRPO trainer wiring for the WarGames Red agent.

Bridges TRL >=0.25 GRPO with the live WarGames environment:

- `LocalGenerationClient` wraps `model.generate` so the episode runner can sample
  one bash command per env step from the policy under training.
- `make_rollout_func` returns a TRL `RolloutFunc` that, for each prompt in a
  group, runs a full env episode and reports per-completion reward via a
  `trajectories` extra field. Requires `use_vllm: true` because TRL only invokes
  rollout_func on the vLLM path.
- `reward_from_rollout` reads `trajectories` from kwargs (TRL forwards rollout
  extra fields as reward kwargs) and aggregates per-step env rewards.
- `build_prompt_dataset` returns a `datasets.Dataset` whose `prompt` column is
  the formatted Red prompt for the initial environment state. Subsequent step
  prompts are built inside the episode loop.
"""

from __future__ import annotations

from typing import Any, Sequence


from training.env_adapter.observation_formatter import build_red_prompt
from training.grpo.config import build_grpo_config
from training.grpo.reward_adapter import aggregate_episode_reward
from training.rollouts.episode_runner import run_episode
from training.rollouts.trajectory import EpisodeTrajectory


class LocalGenerationClient:
    """Generates a single completion per call using the in-training model."""

    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        templated = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(templated, return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0.0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
        import torch

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = output_ids[0, prompt_len:]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True)


def build_prompt_dataset(
    tasks: Sequence[str],
    env_client,
    initial_step: int = 1,
):
    """Build a Dataset with one row per task; `prompt` is the Step-1 Red prompt.

    The env is reset to obtain the initial observation so the model sees a real
    metrics/process snapshot, not just the bare task name.
    """
    from datasets import Dataset

    rows: list[dict[str, Any]] = []
    for task_name in tasks:
        observation = env_client.reset(task_name)
        prompt = build_red_prompt(
            observation=observation,
            task_name=task_name,
            step_num=initial_step,
            attempt_history=[],
        )
        rows.append({"task_name": task_name, "prompt": prompt})
    return Dataset.from_list(rows)


def reward_from_rollout(prompts, completions, **kwargs) -> list[float]:
    """TRL-compatible reward: reads `trajectories` extra field forwarded by the rollout.

    TRL calls reward funcs as `reward_func(prompts=..., completions=..., **reward_kwargs)`
    where reward_kwargs include any extra fields produced by rollout_func, repeated
    per-completion. We expect `trajectories` to be a list of EpisodeTrajectory aligned
    with `completions`.
    """
    trajectories = kwargs.get("trajectories")
    if trajectories is None:
        # Fallback: no episode info available (e.g. non-rollout_func path); reward zero.
        return [0.0 for _ in completions]
    return [
        aggregate_episode_reward(trajectory.rewards, method="sum")
        if isinstance(trajectory, EpisodeTrajectory)
        else 0.0
        for trajectory in trajectories
    ]


def make_rollout_func(llm_client, env_client, max_steps: int, tokenizer):
    """Build a TRL >=0.25 RolloutFunc that runs full env episodes per generation.

    Contract (TRL v0.25): `(prompts, args, processing_class) -> dict` with at minimum
    `prompt_ids`, `completion_ids`, `logprobs`, plus any per-completion extra fields
    (forwarded as kwargs to reward functions).

    For each prompt in the group, we run one independent env episode using the
    current policy. We stitch together the per-step model outputs into a single
    "completion" string the trainer treats as the sample. `trajectories` is the
    extra field carrying per-episode reward info.
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def rollout_func(prompts, *extra_args, **kwargs):
        # TRL v0.25 calls (prompts, args, processing_class).
        # Unsloth's compiled trainer calls (prompts, trainer_self).
        # Resolve processing_class from whichever shape we got.
        processing_class = kwargs.get("processing_class")
        if processing_class is None:
            for cand in extra_args:
                if cand is None:
                    continue
                if hasattr(cand, "__call__") and hasattr(cand, "pad_token_id"):
                    processing_class = cand
                    break
                inner = getattr(cand, "processing_class", None)
                if inner is not None:
                    processing_class = inner
                    break
        if processing_class is None:
            processing_class = tokenizer
        prompt_ids_list: list[list[int]] = []
        completion_ids_list: list[list[int]] = []
        logprobs_list: list[list[float]] = []
        trajectories: list[EpisodeTrajectory] = []

        for prompt in prompts:
            # Trainer passes prompts as already-templated strings (TRL applies chat
            # template before calling rollout_func when prompts are conversational).
            task_name = _extract_task_name(prompt)
            episode = run_episode(
                llm_client=llm_client,
                env_client=env_client,
                task_name=task_name,
                max_steps=max_steps,
            )
            trajectories.append(episode)

            # Encode prompt and a stitched completion for TRL bookkeeping.
            stitched_completion = _stitch_completion(episode)
            prompt_token_ids = processing_class(
                prompt, add_special_tokens=False
            )["input_ids"]
            completion_token_ids = processing_class(
                stitched_completion, add_special_tokens=False
            )["input_ids"]
            if not completion_token_ids:
                completion_token_ids = [pad_id]
            prompt_ids_list.append(list(prompt_token_ids))
            completion_ids_list.append(list(completion_token_ids))
            # TRL accepts `logprobs=None` per-sequence on the non-vLLM tokens path,
            # but the rollout contract requires the field present. Use a zero
            # placeholder per token; TIS correction is bypassed when zeros are sent.
            logprobs_list.append([0.0] * len(completion_token_ids))

        return {
            "prompt_ids": prompt_ids_list,
            "completion_ids": completion_ids_list,
            "logprobs": logprobs_list,
            "trajectories": trajectories,
        }

    return rollout_func


def _extract_task_name(prompt: str) -> str:
    """Recover the task name from a formatted Red prompt.

    `build_red_prompt` always emits a `TASK: <name>` line; this is the most
    robust hook back to the env without changing the dataset shape.
    """
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("TASK:"):
            return stripped[len("TASK:") :].strip()
    raise ValueError("rollout prompt missing required 'TASK:' line")


def _stitch_completion(episode: EpisodeTrajectory) -> str:
    """Join per-step model commands into a single completion string for TRL."""
    if not episode.steps:
        return ""
    return "\n".join(
        f'{{"command":{step.command!r}}}' for step in episode.steps
    )


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
