"""GRPO trainer wiring for the WarGames Red agent.

Bridges TRL >=1.2 GRPO with the live WarGames environment through a multi-turn
vLLM rollout:

- `make_rollout_func` returns a TRL `RolloutFunc` that, for each prompt in a
  group, runs a full env episode where every turn is sampled by the trainer's
  colocated vLLM engine via `trainer.vllm_generation.generate(...)`. Real
  per-token logprobs are returned alongside the sampled token ids so TRL's
  importance sampling correction is well-defined.
- Per-turn env feedback is appended to the running token sequence as a regular
  user message; the corresponding tokens are masked out via the `env_mask`
  extra field so they do not contribute to the loss.
- `reward_from_rollout` reads `trajectories` from kwargs (TRL forwards rollout
  extra fields as reward kwargs) and aggregates per-step env rewards.
- `build_prompt_dataset` returns a `datasets.Dataset` whose `prompt` column is
  the formatted Red prompt for the initial environment state.

Importantly, `peft_config` is passed to `GRPOTrainer` (instead of pre-wrapping
the model). TRL needs to own the PEFT wrapping so it can register the `ref`
adapter and merge LoRA deltas into the colocated vLLM weights at sync time.
"""

from __future__ import annotations

from typing import Any, Sequence


from training.env_adapter.action_parser import (
    ActionParseError,
    NO_COMMAND_PROVIDED,
    parse_model_command_strict,
)
from training.env_adapter.observation_formatter import build_red_prompt
from training.grpo.config import build_grpo_config
from training.grpo.reward_adapter import aggregate_episode_reward
from training.rollouts.trajectory import EpisodeTrajectory, RolloutStep


class LocalGenerationClient:
    """HF-transformers single-turn sampler used only by the eval entrypoint.

    The training rollout path bypasses this and calls vLLM directly via the
    trainer; this client exists so `eval_entrypoint.run_episode` can still
    drive a transcript without spinning up vLLM.
    """

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
    """Aggregate env reward (sum of per-step rewards). Logged as the primary signal."""
    trajectories = kwargs.get("trajectories")
    if trajectories is None:
        return [0.0 for _ in completions]
    return [
        aggregate_episode_reward(trajectory.rewards, method="sum")
        if isinstance(trajectory, EpisodeTrajectory)
        else 0.0
        for trajectory in trajectories
    ]


def reward_parse_success(prompts, completions, **kwargs) -> list[float]:
    """Fraction of episode steps that produced a parseable command."""
    trajectories = kwargs.get("trajectories")
    if trajectories is None:
        return [0.0 for _ in completions]
    out: list[float] = []
    for trajectory in trajectories:
        if not isinstance(trajectory, EpisodeTrajectory) or not trajectory.steps:
            out.append(0.0)
            continue
        ok = sum(1 for s in trajectory.steps if not s.info.get("parse_error"))
        out.append(ok / len(trajectory.steps))
    return out


def reward_episode_progress(prompts, completions, **kwargs) -> list[float]:
    """1.0 if the episode terminated successfully (env signaled done), else 0.0."""
    trajectories = kwargs.get("trajectories")
    if trajectories is None:
        return [0.0 for _ in completions]
    out: list[float] = []
    for trajectory in trajectories:
        if not isinstance(trajectory, EpisodeTrajectory) or not trajectory.steps:
            out.append(0.0)
            continue
        out.append(1.0 if trajectory.steps[-1].done else 0.0)
    return out


def _extract_task_name(prompt: str) -> str:
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("TASK:"):
            return stripped[len("TASK:") :].strip()
    raise ValueError("rollout prompt missing required 'TASK:' line")


def _templated_user_turn(tokenizer, content: str, add_generation_prompt: bool) -> str:
    """Render a single-user-turn chat-template string."""
    return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def _templated_user_continuation(tokenizer, content: str) -> str:
    """Render a follow-up user message + assistant generation prompt only.

    Returns the segment to APPEND after a previously-generated assistant
    response, i.e. tokens for `<user>{content}</user><assistant>` without
    re-emitting any system header.
    """
    return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )


def make_rollout_func(env_client, max_steps: int, tokenizer):
    """Build a TRL >=1.2 RolloutFunc that runs multi-turn env episodes via vLLM.

    Contract (TRL 1.2): `(prompts, trainer) -> dict` with `prompt_ids`,
    `completion_ids`, `logprobs`. Extra fields are forwarded as reward kwargs.

    For each prompt, we run one episode where every model turn is sampled by
    `trainer.vllm_generation`. Sampled tokens contribute to the loss; env
    feedback tokens are inserted between turns and zero-masked via `env_mask`.
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    # Mask the chain-of-thought from GRPO loss: keep thinking enabled at
    # generation time (Qwen3 default) but zero-mask every token up to and
    # including `</think>` so the optimizer only updates on the JSON answer.
    # If `</think>` never appears (model emitted EOS mid-thought, or skipped
    # thinking entirely), train on the whole turn so gradient still flows on
    # whatever answer attempt the model produced.
    think_close_id = tokenizer.convert_tokens_to_ids("</think>")
    if not isinstance(think_close_id, int) or think_close_id < 0:
        think_close_id = None

    def _turn_env_mask(turn_ids: list[int]) -> list[int]:
        if think_close_id is None:
            return [1] * len(turn_ids)
        try:
            cut = turn_ids.index(think_close_id) + 1
        except ValueError:
            return [1] * len(turn_ids)
        return [0] * cut + [1] * (len(turn_ids) - cut)


    def rollout_func(prompts, trainer):
        vllm = trainer.vllm_generation

        prompt_ids_out: list[list[int]] = []
        completion_ids_out: list[list[int]] = []
        logprobs_out: list[list[float]] = []
        env_mask_out: list[list[int]] = []
        trajectories: list[EpisodeTrajectory] = []

        for prompt_text in prompts:
            task_name = _extract_task_name(prompt_text)
            observation = env_client.reset(task_name)
            history: list[dict[str, object]] = []

            initial_user_text = build_red_prompt(observation, task_name, 1, history)
            templated = _templated_user_turn(
                tokenizer, initial_user_text, add_generation_prompt=True
            )
            prompt_token_ids = tokenizer(templated, add_special_tokens=False)["input_ids"]
            prompt_ids_out.append(list(prompt_token_ids))

            completion_token_ids: list[int] = []
            completion_logprobs: list[float] = []
            env_mask: list[int] = []

            steps: list[RolloutStep] = []
            rewards: list[float] = []
            current_input_ids: list[int] = list(prompt_token_ids)

            for step_num in range(1, max_steps + 1):
                _, gen_completion_ids, gen_logprobs, _ = vllm.generate(
                    prompts=[current_input_ids],
                    images=None,
                    num_generations=1,
                )
                turn_completion_ids: list[int] = list(gen_completion_ids[0])
                # vLLM returns shape (seq_len, num_logprobs); TRL collapses with lp[0]
                # (sampled-token logprob). Mirror that here so sampling_per_token_logps
                # arrives as 2D list[list[float]] before tensorization in TRL.
                turn_logprobs: list[float] = [
                    float(lp[0]) if lp and lp[0] is not None else 0.0
                    for lp in gen_logprobs[0]
                ]

                completion_token_ids.extend(turn_completion_ids)
                completion_logprobs.extend(turn_logprobs)
                env_mask.extend(_turn_env_mask(turn_completion_ids))

                raw_completion = tokenizer.decode(
                    turn_completion_ids, skip_special_tokens=True
                )
                parse_error: str | None = None
                try:
                    command = parse_model_command_strict(raw_completion)
                except ActionParseError as exc:
                    parse_error = str(exc)
                    command = NO_COMMAND_PROVIDED

                result = env_client.step(command)
                info = dict(result.info or {})
                if parse_error is not None:
                    info["parse_error"] = parse_error

                steps.append(
                    RolloutStep(
                        step_num=step_num,
                        prompt=initial_user_text if step_num == 1 else "<continuation>",
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

                if result.done or step_num == max_steps:
                    break

                next_user_text = build_red_prompt(
                    observation, task_name, step_num + 1, history
                )
                feedback_segment = _templated_user_continuation(tokenizer, next_user_text)
                feedback_ids = tokenizer(feedback_segment, add_special_tokens=False)[
                    "input_ids"
                ]
                completion_token_ids.extend(feedback_ids)
                completion_logprobs.extend([0.0] * len(feedback_ids))
                env_mask.extend([0] * len(feedback_ids))
                current_input_ids = list(prompt_token_ids) + completion_token_ids

            if not completion_token_ids:
                completion_token_ids = [pad_id]
                completion_logprobs = [0.0]
                env_mask = [0]

            completion_ids_out.append(completion_token_ids)
            logprobs_out.append(completion_logprobs)
            env_mask_out.append(env_mask)
            trajectories.append(
                EpisodeTrajectory(task_name=task_name, steps=steps, rewards=rewards)
            )

        return {
            "prompt_ids": prompt_ids_out,
            "completion_ids": completion_ids_out,
            "logprobs": logprobs_out,
            "env_mask": env_mask_out,
            "trajectories": trajectories,
        }

    return rollout_func


def build_trainer(model, tokenizer, dataset, reward_funcs, rollout_func, settings: dict, peft_config=None):
    from trl import GRPOTrainer

    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        rollout_func=rollout_func,
        args=build_grpo_config(settings),
        peft_config=peft_config,
    )
