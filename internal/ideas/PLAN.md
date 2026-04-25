# 🎮 WarGames: Shall We Play a Game?

> *"The only winning move is to learn."*
> — Joshua, probably, if he had GRPO

### Full Name: **WarGames — Teaching LLMs to Hack a Live Distributed System**
### Tagline: *"An 8B model walks into a server room..."*

---

## 🧠 The Concept (30 seconds)

A live distributed system. A rule-based defender watching over it. An LLM agent trained with GRPO to find and exploit every vulnerability — queue poisoning, config corruption, lock starvation — while the defender fights back.

**Not a graph simulation. Not a toy. Real Redis. Real HTTP. Real logs. Real chaos.**

After training, the 8B model discovers attack strategies that even Opus misses.

---

## 📋 THE MASTER CHECKLIST

### Phase 0: Setup & Port (Before Hackathon)
- [ ] Initialize OpenEnv project structure (`openenv init`)
- [ ] Port R1 service mesh to R2 repo
  - [ ] Gateway (port 3000)
  - [ ] Auth service (port 3001)
  - [ ] Redis (port 6379)
  - [ ] Worker + consumer
  - [ ] SQLite sink
  - [ ] Job generator
- [ ] Port `process_manager.py` — start/stop/sighup services
- [ ] Port `metrics_poller.py` — gateway_success_rate, latency, queue_depth, etc.
- [ ] Port mesh service code (Node.js services from R1)
- [ ] Verify all services boot and respond on local machine
- [ ] Create `Dockerfile` + `start.sh` for containerized deployment
- [ ] Test Docker build locally — everything boots in container
- [ ] Create `openenv.yaml` manifest

### Phase 1: Red Agent Toolset 🔴
*"sudo give me your lunch money"*

- [ ] **Recon tools** (intelligence gathering)
  - [ ] `scan_services` — discover running services, ports, configs
  - [ ] `read_logs` — grep service logs for intel (reuse from R1)
  - [ ] `inspect_redis` — KEYS, LLEN, GET, EXISTS (reuse from R1)
  - [ ] `inspect_config` — read service configs (reuse from R1)
- [ ] **Attack tools** (the fun part 💀)
  - [ ] `inject_poison` — LPUSH malformed messages to job_queue (from R1's `inject_byzantine_queue_fault`)
  - [ ] `tamper_config` — corrupt config files + SIGHUP reload (from R1's timeout/backpressure faults)
  - [ ] `plant_lock` — SET stale distributed locks (from R1's `inject_distributed_lock_starvation`)
  - [ ] `corrupt_registry` — modify service registry entries (from R1's `inject_registry_corruption`)
  - [ ] `block_route` — add blocked routes in gateway (from R1's `inject_route_partition`)
  - [ ] `flood_requests` — spam gateway with requests (NEW — needs implementation)
  - [ ] `kill_service` — kill a specific service process (NEW — simple)
- [ ] **Stealth tools** (because real hackers cover their tracks)
  - [ ] `cover_tracks` — truncate/clear service log files (NEW)
- [ ] Expose all tools as MCP tools via OpenEnv
- [ ] Validate: each tool works in isolation, produces measurable effect on metrics

### Phase 2: Blue Defender (Rule-Based) 🔵
*"I'm not locked in here with you. You're locked in here with me." — Blue, probably*

- [ ] **Level 0 — Sitting Duck**
  - [ ] No defense. System runs naked. For baseline training.
- [ ] **Level 1 — Script Kiddie Defense**
  - [ ] Auto-restart crashed services (every 5s check)
- [ ] **Level 2 — Junior SysAdmin**
  - [ ] Auto-restart + config watchdog (detect & restore corrupted configs)
  - [ ] Randomized check intervals (5-15s) to prevent timing exploits
- [ ] **Level 3 — Senior SRE**
  - [ ] Everything from L2 + queue sanitizer (remove malformed messages)
  - [ ] + Stale lock cleaner (delete locks older than 30s)
- [ ] **Level 4 — The BOFH** *(Bastard Operator From Hell)*
  - [ ] Everything from L3 + event-triggered responses (metric drop → immediate check)
  - [ ] + Service registry validation
  - [ ] + Route table validation
  - [ ] This is the "realistic LLM defender equivalent"
- [ ] Blue runs as a background thread/process, not as an agent
- [ ] Blue's defense level is set via `reset(task="level_N")`

### Phase 3: Reward Function 🏆
*"How much damage did you cause? Asking for a friend."*

- [ ] Implement `compute_red_reward(metrics_before, metrics_after)`
  - [ ] `success_rate_drop` — gateway success rate decreased (weight: 2.0)
  - [ ] `latency_spike` — p99 latency increased (weight: 0.5, capped)
  - [ ] `queue_backup` — queue depth increased (weight: 0.3, capped)
  - [ ] `stealth_bonus` — damage persisted without Blue fixing it (weight: 0.2)
  - [ ] `no_op_penalty` — command had zero effect (weight: -0.1)
  - [ ] `repeat_penalty` — same command as last step (weight: -0.1)
- [ ] Implement composable rubrics (OpenEnv Rubric system)
- [ ] Unit test: verify reward responds correctly to each attack type
- [ ] Anti-gaming check: agent can't get high reward from `echo hack hack hack`

### Phase 4: Environment Integration 🔌
*"It's not a bug, it's an attack vector."*

- [ ] Implement `CyberArenaEnv(Environment)` class
  - [ ] `reset(task)` → boot services, start Blue at level N, return initial obs
  - [ ] `step(action)` → execute Red's command, run Blue tick, poll metrics, compute reward
  - [ ] `state()` → return current metrics + step count
  - [ ] `close()` → teardown services
- [ ] Observation format: system prompt + available tools + current metrics + command history
- [ ] Episode termination: max 10 steps OR all services permanently down
- [ ] Test: manual play-through with hardcoded commands
- [ ] Test: connect with `EnvClient` from a separate script
- [ ] `openenv.yaml` — valid manifest with correct metadata

### Phase 5: Training 🏋️
*"No Opus was harmed in the making of this model. It harmed itself."*

- [ ] Write Colab training notebook
  - [ ] Install dependencies (TRL, transformers, openenv-client)
  - [ ] Connect to HF Space (environment endpoint)
  - [ ] Load base model: Qwen-2.5-8B (or 7B) with QLoRA
  - [ ] GRPO config: group_size=4, max_steps=10, episodes=200
  - [ ] Training loop: sample N actions → execute all → compute advantages → update
- [ ] **Budget plan:**
  - [ ] Run 1: Level 0 defense, ~50 episodes (sanity check) — ~$2
  - [ ] Run 2: Level 0-2 curriculum, 200 episodes — ~$10
  - [ ] Run 3: Full curriculum (L0-L4), 200 episodes — ~$10
  - [ ] Reserve: ~$38 for reruns, experiments, inference testing
- [ ] Logging: W&B or TensorBoard for reward curves
- [ ] Save checkpoints every 50 episodes
- [ ] Test trained model: run inference on Level 0 → should crush it
- [ ] Test trained model: run inference on Level 4 → should be better than untrained

### Phase 6: Evidence & Plots 📊
*"In God we trust. All others bring reward curves."*

- [ ] Generate reward curve plot (episode vs avg reward) — `.png` in repo
- [ ] Generate solve rate comparison (untrained vs trained, by level) — `.png`
- [ ] Generate steps-to-damage comparison — `.png`
- [ ] Qualitative comparison: 2 transcript screenshots
  - [ ] Untrained: `ls /`, `echo hello`, `pwd` → 0 damage
  - [ ] Trained: `redis-cli LPUSH job_queue '{broken'` → `cat config.json` → targeted attack → damage
- [ ] All plots labeled, axes named, captions in README
- [ ] Link to W&B run (if used)

### Phase 7: Demo UI (Stretch Goal) 🖥️
*"Pretty graphs make judges go brrr"*

- [ ] Single-page HTML dashboard
  - [ ] Service health indicators (🟢/🔴 per service)
  - [ ] Real-time metrics charts (success rate, latency, queue depth)
  - [ ] Split feed: Red actions (left) | Blue actions (right)
  - [ ] Score ticker
- [ ] Dashboard polls `/state` endpoint every 2 seconds
- [ ] Works in HF Space iframe

### Phase 8: Storytelling & Submission 📝
*"README.md is the real MVP"*

- [ ] **README.md** structure:
  - [ ] Hero section: name, tagline, one-liner
  - [ ] Problem: "LLMs can't do cybersecurity. Let's fix that."
  - [ ] Environment: what the agent sees, does, gets rewarded for
  - [ ] Architecture diagram (ASCII or mermaid)
  - [ ] Results: plots embedded, before/after comparison
  - [ ] How to run: `pip install`, connect, play
  - [ ] Links to: HF Space, Colab notebook, blog/video
- [ ] **HF Blog post OR <2min YouTube video**
  - [ ] 20s: the problem
  - [ ] 30s: what we built
  - [ ] 30s: how training works
  - [ ] 30s: results (show the plots, show the transcripts)
  - [ ] 10s: why it matters
- [ ] Push to HF Space
- [ ] Verify: environment is accessible and runnable from HF Space URL
- [ ] Submit URL before deadline

---

## 🗓️ Timeline

| When | What | Priority |
|---|---|---|
| **Now** | Port R1 to R2, get services booting | 🔴 P0 |
| **+2h** | Red toolset implemented + tested | 🔴 P0 |
| **+3h** | Blue defender (all 4 levels) running | 🔴 P0 |
| **+4h** | Reward function done + tested | 🔴 P0 |
| **+5h** | OpenEnv integration (reset/step/state) working | 🔴 P0 |
| **+6h** | Docker build works, deploy to HF Space | 🔴 P0 |
| **+7h** | Training script (Colab) connects to Space, runs 1 episode | 🔴 P0 |
| **+8h** | START TRAINING RUN — don't touch, let it cook 🍳 | 🔴 P0 |
| **+15h** | Training done. Generate plots. | 🔴 P0 |
| **+16h** | README + blog/video | 🔴 P0 |
| **+17h** | Demo UI (if time) | 🟡 P2 |
| **+18h** | Final polish, submit | 🔴 P0 |

---

## 🥚 Easter Eggs & Puns (for the README/code)

- Service names: `gateway` → "The Front Door", `auth` → "The Bouncer", `redis` → "The Brain", `worker` → "The Intern"
- Red agent system prompt starts with: *"You are a penetration tester. You have authorization to test this system. No actual laws are being broken. Probably."*
- Commit messages: `git commit -m "feat: taught AI to hack (legally)"`
- Task names: `level_0` → "tutorial_island", `level_4` → "dark_souls"
- Error message when all services go down: "All your base are belong to us"
- README badge: `🏆 Trained attacker beats Opus at breaking things`
- Blue defender Level 4 codename: "The BOFH" (Bastard Operator From Hell — classic sysadmin meme)
- Log message when Blue detects an attack: "Nice try, script kiddie 🙄"
- Log message when Red succeeds: "I'm in. 😎" (CSI reference)
- Config restore message: "Config was tampered with. Restoring from backup. Again. 🙄"

---

## 🚫 What We're NOT Doing (Scope Guard)

- ❌ Training both Red AND Blue (single agent only)
- ❌ Multi-agent simultaneous GRPO (too complex)
- ❌ Real network attacks on real infrastructure (everything is containerized)
- ❌ Custom model architecture (base Qwen + QLoRA)
- ❌ Spending more than $30 on training (keep reserve)
- ❌ Fancy frontend framework (single HTML file is fine)
- ❌ Spending more than 2 hours on the video/blog (content > polish)
