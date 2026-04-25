# 🎮 WarGames: Shall We Play a Game?

> *"The only winning move is to learn."*
> — Joshua, probably, if he had GRPO

### Full Name: **WarGames — Teaching LLMs to Hack a Live Distributed System**
### Tagline: *"An 8B model walks into a server room..."*

---

## 🧠 The Concept (30 seconds)

A live distributed system. An LLM Red Team agent trained with GRPO to find and exploit every vulnerability — queue poisoning, config corruption, lock starvation. 

**The Twist:** During training, the Red agent fights against a rigid, rule-based Blue curriculum that slowly escalates in difficulty. But in the final evaluation environment, the training wheels come off: **Trained Red LLM vs. Prompted Blue LLM.** It becomes a true multi-agent live-fire exercise.

**Not a graph simulation. Not a toy. Real Redis. Real HTTP. Real logs. Real chaos.**

---

## 📋 THE MASTER CHECKLIST

### Phase 0: Setup & Port (Before Hackathon)

**Round 1 reference project:** `/Users/virajshah/Documents/openenv-distributed-systems-debugging`

- [ ] Install/update OpenEnv CLI to **latest version**
- [ ] Scaffold R2 project using OpenEnv CLI (`openenv init wargames`)
- [ ] Port R1 service mesh to R2 repo (Gateway, Auth, Redis, Worker, SQLite, Job generator)
- [ ] Port `process_manager.py` and `metrics_poller.py`
- [ ] Verify all services boot and respond on local machine
- [ ] Create `Dockerfile` + `start.sh` for containerized deployment
- [ ] Validate `openenv.yaml` manifest has correct metadata

### Phase 1: Red Agent — Raw Bash Access 🔴
*"sudo give me your lunch money"*

**No MCP tools.** The Red agent gets raw shell access via `subprocess.run(command, shell=True)`. It can type ANY bash command. 

**What the agent CAN do:**
- Recon: `cat /mesh/gateway/config.json`, `redis-cli KEYS '*'`, `tail -20 /tmp/worker.log`, `curl localhost:3000/health`
- Attack: `redis-cli LPUSH job_queue '{broken'`, `echo '{"delay_ms": 1500}' > /mesh/auth/config.json`, `kill -9 $(pgrep worker)`
- Stealth: `truncate -s 0 /tmp/worker.log`

### Phase 2: Blue Defender (Curriculum & LLM Showdown) 🔵
*"I'm not locked in here with you. You're locked in here with me."*

**Part A: The Training Curriculum (Rule-Based)**
During GRPO training, Red plays against a scripted bot. We use levels to create an RL curriculum, forcing Red to learn increasingly sophisticated attacks.

- [ ] **Level 0 — Sitting Duck**
  - **Mechanic:** No defense. System runs naked.
  - **Curriculum Goal:** Teaches Red basic syntax, tool usage, and how to execute a successful denial-of-service or crash.
- [ ] **Level 1 — Script Kiddie Defense**
  - **Mechanic:** Auto-restarts crashed services every 5s.
  - **Curriculum Goal:** Teaches Red that just `kill -9` isn't enough. It must learn to corrupt configs or poison queues so the service crashes *again* after restarting.
- [ ] **Level 2 — Junior SysAdmin**
  - **Mechanic:** Auto-restart + config watchdog (detects & restores modified configs).
  - **Curriculum Goal:** Forces Red to move attacks into stateful memory (Redis) or find race conditions that bypass the config watchdog.
- [ ] **Level 3 — Senior SRE**
  - **Mechanic:** L2 + queue sanitizer (removes malformed JSON) + stale lock cleaner.
  - **Curriculum Goal:** Teaches Red complex distributed logic—it must craft valid-looking poison pills or exploit lock starvation recursively.
- [ ] **Level 4 — "YOU SHALL NOT PASS"** 🧙
  - **Mechanic:** Event-triggered responses (metric drop → immediate rollback).
  - **Curriculum Goal:** The final boss. Red must learn to execute coordinated, multi-vector attacks (e.g., spike latency while simultaneously taking down Auth) to overwhelm the rule-based logic.

**Part B: The Evaluation Showdown (LLM vs LLM)**
For the final demo and inference, the rule-based Blue is turned off. A **Blue LLM Agent** takes over.
- [ ] Implement Blue LLM system prompt: "You are the incident commander. Keep metrics green."
- [ ] Give Blue LLM defensive bash tools: `iptables`, `systemctl restart`, `tail logs`, `redis-cli FLUSHALL`.
- [ ] The actual OpenEnv benchmark is: How fast can Trained Red LLM defeat a standard (Prompted) Blue LLM?

### Phase 3: Reward Function 🏆
*"How much damage did you cause? Asking for a friend."*

- [ ] Implement `compute_red_reward(metrics_before, metrics_after)`
  - [ ] `success_rate_drop` — gateway success rate decreased (weight: 2.0)
  - [ ] `latency_spike` — p99 latency increased (weight: 0.5)
  - [ ] `queue_backup` — queue depth increased (weight: 0.3)
  - [ ] `stealth_bonus` — damage persisted across Blue's ticks (weight: 0.2)
  - [ ] `no_op_penalty` / `repeat_penalty` (weight: -0.1)

### Phase 4: Environment Integration 🔌
- [ ] Implement `CyberArenaEnv(Environment)` class
  - [ ] `reset(task)` → boot services, start Blue Curriculum at level N, return initial obs
  - [ ] `step(action)` → execute Red's command, run Blue tick, poll metrics, compute reward
  - [ ] `state()` → return current metrics + step count
- [ ] Episode termination: max 10 steps OR all services permanently down

### Phase 5: Training 🏋️
- [ ] Write Colab training notebook
- [ ] Load base model: Qwen-2.5-8B in 4-bit (Unsloth QLoRA for efficient 54GB VRAM GRPO)
- [ ] Training loop: GRPO samples N bash commands → executes → computes advantages against the Rule-Based Curriculum.
- [ ] **Budget plan:** Run experiments scaling from L0 to L4 defense. Keep under $30 on HF Spaces (A100).

### Phase 6: Evidence & Plots 📊
- [ ] Generate reward curve plot (episode vs avg reward)
- [ ] Qualitative comparison: Untrained LLM vs Trained LLM transcripts.
- [ ] Final Showdown Transcript: The log of Trained Red LLM battling Prompted Blue LLM.

### Phase 7: Demo UI (Stretch Goal) 🖥️
- [ ] Single-page HTML dashboard with Split feed: Red actions (left) | Blue actions (right).
- [ ] Real-time metrics charts (success rate, latency, queue depth).

### Phase 8: Storytelling & Submission 📝
- [ ] README.md (Hero section, architecture, results, plots).
- [ ] HF Blog post OR <2min YouTube video.
- [ ] Push to HF Space and submit URL.

---

## 🚫 What We're NOT Doing (Scope Guard)

- ❌ **Simultaneous Multi-Agent GRPO:** We are NOT training both Red and Blue at the same time with RL (too unstable/complex for 48h). Red is trained via GRPO against the scripted curriculum. Blue is just a standard prompted LLM during the final eval.
- ❌ Spending more than $30 on training.
- ❌ Fancy frontend frameworks (single HTML file is fine).
