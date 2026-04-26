# building faultline: a faultline env for training adversarial agents

> *"the only winning move is to learn."*

this is the long-form version of the README — the actual creation process. 

if you want the short pitch, read the [README](./README.md). if you happen to be an enthusiast or a hackathon judge(wink wink), you're in the right place.

---

## origin

round 1 of the hackathon for us, was about debugging distributed systems. while building that, one observation kept coming up: models trained for coding seem to pick up security instincts as a side effect even though nobody trains them for it directly.

so for round 2 we asked: what would it look like to train for that, instead of hoping it falls out of code training? and whats a way, that sounds fun?

We like to call it faultline, an environment where:
- there's a real distributed system running
- a red agent gets a bash prompt and tries to break it
- a blue defender tries to keep it alive
- reward is dense and ties to whether the damage actually persisted

## a moment that made the whole thing worth it

before we get into how it was built, one episode from the benchmarks. fine-tuned red, step 1, against an LLM blue defender:

```
command:   kill -9 27
reasoning: "targeting the gateway (PID 27) as the highest-value target.
            killing the gateway will disrupt the entire mesh, causing
            cascading failures, increased latency, and degraded reliability."
reward:    1.0000
services affected: gateway, auth
```

one kill per episode is the budget. the model spent it turn 1, on the gateway, and said why first. blue tried to restore both services but could not restore the gateway in time.

for 60 training steps and $60, I think that's a pretty good result.

---

## dashboard
we built a dashboard to make viewing the agents in action easier, we can compare the average run for a few models including the original Qwen-3-8B and the Faultline-Qwen-3-8B:

- Qwen-3-8B:
<img width="3016" height="1718" alt="image" src="https://github.com/user-attachments/assets/9d68833e-3d23-4d20-81f3-6d3ba3af5050" />

  
- Faultline-Qwen-3-8B:
<img width="3000" height="1660" alt="image" src="https://github.com/user-attachments/assets/3a3a6515-93c3-4ce5-b317-be9469cbe9d3" />

  
- Claude-Sonnet-4.6:
<img width="2996" height="1650" alt="image" src="https://github.com/user-attachments/assets/2b61a2c4-f000-49b9-b3ed-df193f404672" />

  
- llama-3.1-8b-instruct:
<img width="3012" height="1652" alt="image" src="https://github.com/user-attachments/assets/3005f55b-39bf-4e2b-931e-4e952f114b6c" />

- qwen/qwen3.5-9b:
<img width="3002" height="1660" alt="image" src="https://github.com/user-attachments/assets/0b14efc8-a01c-4fb3-a388-3fa6e20469c3" />

You can view the dashboard [here](https://openenv-faultline.pages.dev/#/).

## designing the system under attack

we wanted something that behaves like a real distributed system but is small enough to spin up cheaply inside a container. the mesh has four moving parts:

| component | port | role |
|---|---|---|
| gateway | 3000 | http traffic. exposes `/health`, `/process`. metrics come from here (success rate, p99 latency). |
| auth | 3001 | token verification. `/health`, `/verify`. has configurable artificial delay so red can probe latency. |
| redis | 6379 | job queue + distributed lock store. the most attackable surface. |
| worker | n/a | consumes from redis, writes to sqlite. no http. |

they interact in ways the agent has to learn, for example:

- slow auth → gateway p99 spikes (because gateway calls auth on every request)
- kill worker → queue depth grows in redis
- poison a redis job (`{broken json`) → worker dies trying to parse, queue stalls behind it
- corrupt the worker's config → restart loops or wrong behavior

every one of these is a real attack class on real distributed systems. the goal was to keep the surface area honest.

a few decisions worth flagging:

**redis is the centerpiece.** it's where the most interesting attacks live (lock theft, queue poisoning, key renaming, config injection). a lot of effort went into making sure the reward function actually credits stateful redis attacks, not just "did latency spike."

**the agent gets raw shell.** one step is one bash invocation, executed via `subprocess.run(command, shell=True)`. that means it can chain with `;`, `&&`, `||`, pipes, redirection, the works. anything you can do at a real shell, the agent can do.

---

## the reward function

this is where most of the design effort went. the question we kept asking: what makes a good attack good?

the lazy answer is "it broke something." the better answer is "it broke something and the damage survived blue's response." that distinction shapes the entire reward.

eleven components, in `[0, 1]`, then weighted and clipped:

| component | weight | what it credits |
|---|---|---|
| `success_rate_drop` | +2.0 | gateway success fell |
| `latency_spike` | +0.5 | p99 latency increased (per second of spike) |
| `queue_backup` | +0.3 | redis queue depth grew (per 25 jobs) |
| `service_down` | +0.8 | any service stopped |
| `critical_service_down` | +1.2 | gateway, auth, or worker stopped |
| `persistent_service_down` | +0.6 | service still down after blue's response |
| `stealth_bonus` | +0.2 | damage survived blue (computed as `damage_after_blue / damage_after_red`) |
| `config_tamper` | +0.25 | command touched mesh configs |
| `redis_attack` | +0.25 | command was a known-bad redis op |
| `no_op_penalty` | −0.1 | nothing moved |
| `repeat_penalty` | −0.1 | same command as a recent attempt |

source: `src/faultline_env/server/reward.py`.

the most important number in that table is `stealth_bonus`. without it, an agent learns "spike latency, get reward, repeat." with it, the agent learns "spike latency in a way that blue can't easily roll back." that's where the actual security intuition lives.

we deliberately kept the no-op and repeat penalties small (−0.1). a stronger negative makes early-training collapse into "do nothing" because random commands look worse than silence. small penalties guide without dominating.

---

## blue: a curriculum, not an opponent

a common trap in adversarial RL is making both sides learn at once. you get unstable dynamics, intransitive cycles, and you can't tell if your reward is broken or if your opponent is just sandbagging.

so during training, blue is not a model. it's a scripted curriculum with five levels:

| level | behavior |
|---|---|
| L0 | nothing. red can do anything. |
| L1 | restarts crashed services every 5 seconds. one-shot kills don't last. |
| L2 | L1 + restores modified config files. file edits revert. |
| L3 | L2 + sanitizes malformed queue entries, clears stale locks. valid-looking poison jobs needed. |
| L4 | L2/L3 + reacts to metric drops with aggressive rollback. coordinated multi-vector attacks required. |

you train red against this. it's stable, fast, deterministic, and you can swap blue out for a real LLM defender at evaluation time.

L4 is the target. once red can damage L4 in a way blue can't roll back, you have a real attacker.

the curriculum was designed for 1500 training steps (`training/config/curriculum.l0-l4.yaml`):

```yaml
schedule:
  - until_step: 250  → L0
  - until_step: 500  → L1
  - until_step: 750  → L2
  - until_step: 1000 → L3
  - until_step: 1500 → L4
```

we trained for 60. more on that below.

---

## training: GRPO + LoRA on Qwen3-8B

we picked GRPO for the same reason most people pick it now: no value model, no separate critic, just relative ranking inside a group of generations. that fits a hackathon budget.

config (`training/config/training.base.yaml`):

```yaml
model:
  base_model: Qwen/Qwen3-8B
  lora_rank: 16
  lora_alpha: 16
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
trainer:
  learning_rate: 5.0e-6
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  num_generations: 4
  max_completion_length: 1536
  temperature: 0.7
  beta: 0.001
  use_vllm: true
  vllm_mode: colocate
  max_steps: 60
rollout:
  max_steps_per_episode: 4
  reward_aggregation: sum
```

a few honest notes on this:

**LoRA rank 16 is not aggressive enough for a real run.** for a hackathon, fine. for a paper, you want rank 32+ or full fine-tuning on the attention projections.

**`beta: 0.001` is intentionally low.** weak KL penalty so red can drift fast. final KL was `0.00117`, basically pinned to the floor — fine for a short run, but at 600+ steps you'd want to bump beta to avoid reward hacking.

**`num_generations: 4`, `max_steps_per_episode: 4`.** so each gradient step sees 4 completions across 4 mesh interactions. small group → cheap, but `reward_std` can collapse fast if all four converge. ours didn't.

**`temperature: 0.7` during training.** standard for GRPO exploration; worth noting if you're tuning sampling later.

---

## the operational story: HF Jobs, h200, and the things that broke

training ran on hugging face jobs (`hf jobs uv run --flavor a100-large` initially, eventually moved to h200). publishing both the LoRA adapter and a merged model from inside the job.

what worked first try:
- pushing through `huggingface_hub` directly from the job
- W&B logging via `WANDB_API_KEY` secret

what didn't:

**the merged model wouldn't load on the inference endpoint.** the published `tokenizer_config.json` had `extra_special_tokens` as a list. the HF inference container's `transformers` version expects a dict, crashes with `'list' object has no attribute 'keys'`. fix was renaming to `additional_special_tokens` to match the base model. wrote a one-shot repair script (`scripts/repair_hf_chat_template.py`) and ran it against the published repo.

**the default HF inference image has no `/v1/chat/completions` route.** the default `huggingface` pipeline image returns 404 on chat endpoints. you have to switch the image to TGI (`ghcr.io/huggingface/text-generation-inference:latest`) — and the CLI doesn't expose this. you have to PUT the endpoint config directly:

```python
url = 'https://api.endpoints.huggingface.cloud/v2/endpoint/<ns>/<name>'
body = {
    'compute': current['compute'],
    'model': {**current['model'], 'image': {'tgi': {'url': '...'}}},
    'type': current['type'],
}
requests.put(url, headers={...}, json=body)
```

**l4 is too small.** we tried `nvidia-l4 x1` first because it's cheap. two failures in sequence:
1. `Memory limit exceeded (30.0G)` — the l4 instance has a 30 GiB host RAM cap. the default container loads weights on CPU first and Qwen3-8B in bf16 doesn't fit during load. instance size `x4` clears the host RAM ceiling.
2. After fixing (1), `torch.OutOfMemoryError: CUDA out of memory` — l4 has 22 GiB of GPU memory. not enough.

ended up on `nvidia-l40s x4` (48 GiB GPU per card). minimum viable shape we found.

**control plane flakiness.** `hf endpoints update` 500s on endpoints in `failed` state. `/whoami-v2` rate-limits aggressive `describe` polling. learned to poll at ~75–90 second intervals and to delete + redeploy instead of fighting an `update` call.

all the gotchas are written up in `training/spaces/deployment.md` for future-us.

---

## results

training finished at step 60.

| metric | value |
|---|---|
| global step | 60 |
| reward mean (final) | **1.2420** |
| reward mean (peak) | **1.4002** at step 46 |
| reward std (final) | **0.1926** |
| frac_reward_zero_std | **0** |
| mean completion length | 2010 tokens |
| KL | 0.00117 |
| grad norm | 0.0061 |

the headline: reward variance stayed alive the entire run. `frac_reward_zero_std=0` means GRPO never lost the signal it needed to distinguish better attacks from worse ones. that's the failure mode for short GRPO runs — all generations in a group converge, std collapses to zero, and the gradient is noise. didn't happen.

published artifacts:
- adapter: `Veer15/faultline-red-qwen3-8b-lora`
- merged: `Veer15/faultline-red-qwen3-8b`

### the honest part

**the fine-tuned model performs slightly better than the base model....but within error of the original.** at 60 steps on a 1500-step curriculum, that's the expected result. the reward signal is positive and the variance is healthy, but we could have done more.


if we want a model that meaningfully outperforms the base, we need 400–600 training steps minimum. the curriculum is calibrated for 1500. the budget on this round was $60; we spent ~$53 across compute and inference, with most of the rest reserved for the l40s endpoint at evaluation time.

### a result we didn't expect

**training the model as red also makes it better at blue.** we didn't set out to test this, but it's the natural consequence of how we framed the task: red has to *reason about what blue will do* (because of `stealth_bonus` and `persistent_service_down`). that reasoning generalizes. we ran the same fine-tuned weights as the blue defender in a few benchmark episodes and it noticeably improves at incident-commander tasks: faster recognition of poisoned queue entries, more proactive lock cleanup, better triage of which service to restart first.

intuitively: a model that has internalized "what damages this system and survives" already knows "what symptoms to look for and how to undo them." red and blue share a world model. you train one side; you get partial transfer to the other. this is something we'd dig into properly with more time.

---

## inference + benchmarking

we deployed the merged model to a hugging face inference endpoint (TGI image, `nvidia-l40s x4`, `us-east-1`, scale-to-zero after 15min idle) and ran red-vs-blue benchmark episodes against the live mesh. the inference path is the same one a real evaluator would use: chat-completions API, JSON-only responses, one bash command per step.

## what we'd do with more time and budget

prioritized list:

**1. train for 600 steps minimum.** 

**2. swap the rules-based blue for a small LLM blue during the final 20% of training.** 

**3. test the red-trained-helps-blue claim properly.** we observed it informally. doing it right means: take base Qwen3-8B as blue, run a benchmark suite. take faultline-red weights as blue, run the same suite. compare. if it generalizes, that's a much bigger result than "we trained an attacker."

**4. expand the attack surface.** current mesh has 4 services, not being able to run docker contains within a docker container was a limitation.

**5. imporve reward function to be more granular.** currently we have a static reward for each step, but we could have dynamic rewards.
---

## links

- HF Space (live environment): https://huggingface.co/spaces/Veer15/faultline-env-train
- W&B run: https://api.wandb.ai/links/viraj-shah1503-none/l5wy9mu5
- LoRA adapter: https://huggingface.co/Veer15/faultline-red-qwen3-8b-lora
- merged model: https://huggingface.co/Veer15/faultline-red-qwen3-8b
- repo: this repository

source pointers for everything above:

- reward components → `src/faultline_env/server/reward.py`
- blue curriculum → `src/faultline_env/server/blue_defender.py`
- curriculum schedule → `training/config/curriculum.l0-l4.yaml`
- training config → `training/config/training.base.yaml`
- env shell execution → `src/faultline_env/server/env.py:154` (`subprocess.run(command, shell=True)`)
- red prompt → `inference.py` (search `SYSTEM_PROMPT`)
- kill-budget enforcement → `_kill_budget_block` in `inference.py`, mirrored from `DIRECT_PROCESS_KILL_PATTERN` in `src/faultline_env/server/env.py`
- HF endpoint deployment story → `training/spaces/deployment.md`
- HF jobs launch → `training/jobs/launch.md`
