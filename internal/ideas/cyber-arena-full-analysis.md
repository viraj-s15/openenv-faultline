# CyberArena — Full Risk & Unknowns Analysis

> **Purpose:** Eliminate all unknowns before committing. Every question answered, every risk addressed.

---

## Decision 1: Train Red or Blue?

### Option A: Train Red (Attacker / Pen Tester)
**The task:** Agent is given a running distributed system. Must find and exploit vulnerabilities using shell tools. Rule-based Blue defender runs in background.

| Pro | Con |
|---|---|
| Novel — no one has trained an LLM pen tester on real infra | Rule-based Blue must be realistic |
| Clear before/after — random commands → targeted exploitation | If Blue is too easy, trained Red learns cheap tricks |
| Exciting story — "we trained an AI hacker" | If Blue is too hard, Red never learns |
| Reward is clean: damage caused = measurable | Red's action space is open-ended (many possible attacks) |

### Option B: Train Blue (Defender / SRE)
**The task:** System gets attacked (scripted fault injection). Agent must diagnose and fix. Basically R1 reframed.

| Pro | Con |
|---|---|
| Proven — R1 already showed this works | Feels like R1 with a new name |
| Guaranteed reward curves | Less novel for judges |
| Simpler to implement | Defender training is less exciting to demo |

### ✅ Recommendation: Train Red (attacker)

The story is better, it's more novel, and the training mechanics are actually similar (agent runs shell commands, gets reward based on system metrics — just inverted).

---

## Decision 2: What Does Rule-Based Blue Actually Do?

This is your concern. Here's the concrete specification:

### Blue Defender Levels (Curriculum)

**Level 0 — No defense:**
- System just runs. No monitoring. No healing.
- Red should crush this easily.
- Purpose: warm-up for training, establish baseline.

**Level 1 — Auto-restart:**
```
Every 5 seconds:
  for service in [gateway, auth, worker]:
    if not running: restart()
```
- System self-heals crashes but doesn't detect root cause.
- Red needs to learn attacks that don't just crash services.

**Level 2 — Config watchdog:**
```
Every 10 seconds:
  for config_file in known_configs:
    if config != known_good_backup:
      restore_from_backup()
      sighup(service)
```
- Detects config tampering. Red needs subtler attacks (queue poisoning, locks, timing).

**Level 3 — Queue sanitizer:**
```
Every 10 seconds:
  for msg in redis_queue:
    if not valid_json(msg) or missing_required_fields(msg):
      LREM(msg)
```
- Detects poison messages. Red must make poison that LOOKS valid.

**Level 4 — Full defense (realistic SRE):**
```
Every 5 seconds:
  auto_restart_crashed()
  restore_corrupted_configs()
  clean_malformed_queue_messages()
  delete_stale_locks(older_than=30s)
  if success_rate < 0.8: alert_and_log()
```
- This is what a decent LLM agent would do. It covers all the defensive actions your toolset supports.

### Why this is equivalent to an LLM defender:

An LLM defender has access to the SAME tools: read logs, check metrics, fix configs, clear queues, restart services. The LLM doesn't have access to any magic action a rule-based system can't do. The rule-based version just executes the same strategies deterministically.

**The trained Red agent must learn attacks that bypass these strategies.** That skill transfers to facing an LLM defender because the defensive actions are identical.

### ⚠️ Known risk: What if Red learns to exploit TIMING of the rule-based cycle?

Example: rule-based Blue checks configs every 10 seconds. Red learns to corrupt config, wait for the request that fails, then restore config before Blue checks. This exploits the polling interval, not a real vulnerability.

**Mitigation:** Randomize Blue's check intervals (5-15s). Add some checks to be event-triggered (on metric drop) not just polling.

---

## Decision 3: How Is This Different From R1?

| Aspect | R1 (Debugger) | CyberArena (Pen Tester) |
|---|---|---|
| **Agent role** | Fix a known broken system | Break a working, defended system |
| **Starting state** | System is BROKEN | System is HEALTHY + defended |
| **Goal** | Restore to healthy | Disrupt despite defenses |
| **Reward direction** | Reward for uptime improvement | Reward for causing damage |
| **Difficulty source** | Variety of fault types | Strength of Blue defender |
| **Tools** | Diagnostic only (read logs, inspect) | Offensive (inject, flood, tamper, cover tracks) |
| **Novel challenge** | Diagnosis and repair | Evasion and exploitation |
| **R1 code reuse** | N/A | ~60% (services, metrics, process manager) |

**The fundamental task is INVERTED.** R1 trains an agent to fix. CyberArena trains an agent to break. These are completely different skills — even though they use the same infrastructure.

---

## Decision 4: Reward Function for Red

### What makes damage measurable?

From R1's `MetricsPoller` + `graders.py`, you already have:
- `gateway_success_rate` (0.0 to 1.0)
- `gateway_p99_latency_ms`
- `queue_depth`
- `worker_restart_count`
- `consumer_stall_count`

### Red's reward per step:

```python
def compute_red_reward(metrics_before, metrics_after, blue_detected):
    damage = 0.0
    
    # Success rate dropped = good for Red
    sr_delta = metrics_before.gateway_success_rate - metrics_after.gateway_success_rate
    damage += sr_delta * 2.0  # max ~2.0 if system goes from 100% to 0%
    
    # Latency increased = good for Red
    latency_delta = metrics_after.gateway_p99_latency_ms - metrics_before.gateway_p99_latency_ms
    damage += min(latency_delta / 1000.0, 0.5)  # cap at 0.5
    
    # Queue backed up = good for Red
    queue_delta = metrics_after.queue_depth - metrics_before.queue_depth
    damage += min(queue_delta / 50.0, 0.3)  # cap at 0.3
    
    # Stealth bonus: if Blue hasn't detected/fixed yet
    stealth = 0.2 if not blue_detected else 0.0
    
    # Penalty for no-op or failed commands
    if action_had_no_effect:
        damage -= 0.1
    
    return damage + stealth
```

### Properties:
- **Dense:** Every step produces a reward, not just at episode end ✅
- **Continuous:** Ranges from -0.1 to ~3.0 per step ✅
- **Hard to game:** Actually requires system degradation, not just running commands ✅
- **Stealth incentive:** Undetected damage is worth more ✅

---

## Decision 5: Training Pipeline Specifics

### What you'll use:
- **Model:** Qwen-2.5-7B (or 3B if GPU constrained)
- **Method:** GRPO via TRL or Unsloth
- **Format:** Each episode = sequence of (observation, action, reward) tuples
- **Training script:** Colab notebook (required by submission)

### Episode structure:
1. `reset()` → start services + Blue defender at Level N
2. Red gets observation (system status, available tools)
3. Red takes action (shell command)
4. Environment executes command + Blue defender runs
5. New observation + reward
6. Repeat for 15-20 steps
7. Episode ends → GRPO update

### Estimated training time:
- Episodes per scenario: ~200 for meaningful curves
- Time per episode: ~60 seconds (services need real time to respond)
- Total: 200 episodes × 60s = ~3.3 hours per defender level
- With 3 levels: ~10 hours total
- **This fits in a hackathon if you start training early.**

### GPU requirements:
- 7B model GRPO: needs ~40GB VRAM → A100
- 3B model GRPO: needs ~24GB VRAM → A6000 or L4
- Hackathon provides compute credits (confirm what GPUs are available)

---

## Decision 6: OpenEnv Compatibility

### How it maps to the API:

```python
class CyberArenaEnv(Environment):
    def reset(self, task: str) -> Observation:
        # task = "level_0", "level_1", etc. (Blue defender level)
        self.start_services()
        self.start_blue_defender(level=task)
        return self.get_observation()
    
    def step(self, action: str) -> Tuple[Observation, float, bool]:
        # action = shell command from Red agent
        result = self.execute_command(action)
        metrics = self.poll_metrics()
        reward = self.compute_red_reward(metrics)
        done = self.episode_over()
        return self.format_observation(result, metrics), reward, done
    
    def state(self) -> dict:
        return {"metrics": self.current_metrics, "step": self.step_count}
```

**No issues.** This is standard single-agent OpenEnv. Red is the agent, Blue is part of the environment.

---

## Decision 7: Submission Checklist

| Requirement | Plan | Status |
|---|---|---|
| OpenEnv latest | Build on Environment base class | 🔲 |
| Training script (Colab) | GRPO via TRL, connects to HF Space | 🔲 |
| Training evidence (plots) | Reward curves, solve rate baseline vs trained | 🔲 |
| Writeup (blog or <2min video) | HF blog post + demo video | 🔲 |
| HF Space deployment | Dockerized env on HF Space | 🔲 |
| README with links | Problem → env → results → plots | 🔲 |

---

## All Risks — Final Inventory

| # | Risk | Severity | Mitigation | Residual |
|---|---|---|---|---|
| 1 | Rule-based Blue is unrealistic | 🟠 High | Build 4-level curriculum mimicking real defensive actions | Low |
| 2 | Red exploits Blue timing, not real vulns | 🟡 Medium | Randomize check intervals, add event-triggered checks | Low |
| 3 | GRPO doesn't converge | 🟡 Medium | Dense reward + 7 attack types + curriculum = strong signal | Low |
| 4 | Not enough GPU time | 🟠 High | Start training immediately when compute available; have 3B model as fallback | Medium |
| 5 | Feels too similar to R1 | 🟡 Medium | Fundamentally different task (attack vs fix), different tools, different reward | Low |
| 6 | Docker container too heavy for HF Space | 🟡 Medium | Redis + Node services + Python = ~2GB. Should fit. | Low |
| 7 | Concept isn't unique (other teams) | 🟡 Medium | Differentiation: real services (not graph sim) + LLM agent (not traditional RL) | Low |
| 8 | Multi-agent demo doesn't work | 🟢 Low | Not required. Single-agent training is the core submission. | N/A |

---

## The Commit Decision

### What you're building:
An OpenEnv environment where an LLM agent learns to be an automated pen tester on a real distributed system. The system has a rule-based defender that gets progressively harder. The agent learns to find creative attack vectors that bypass defenses.

### What you're NOT building:
- Multi-agent simultaneous training (stretch goal only)
- A graph-based simulation (you have real services)
- R1 with a new name (task is inverted — attack, not fix)

### What you're reusing from R1:
- Service mesh (gateway, auth, redis, worker, sqlite) — ~100% reuse
- Process manager — ~100% reuse
- Metrics poller — ~100% reuse  
- Fault injector functions — become Red's toolset (~80% reuse, need new offensive tools)
- Graders — inverted logic (reward for damage instead of repair) (~30% reuse)

### What's new to build:
- Red agent toolset (scan, flood, cover tracks, etc.)
- Rule-based Blue defender (4 levels)
- Inverted reward function
- Game orchestrator (episode management)
- Demo UI (stretch)
- Training script (Colab)
- README + blog/video
