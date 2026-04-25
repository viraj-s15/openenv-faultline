# CyberArena: Red Team vs Blue Team on a Live Distributed System

> **Theme:** #1 Multi-Agent Interactions (+ #4 Self-Improvement via self-play)
> **One-liner:** Hackerman vs The Admin — two LLM agents wage cybersecurity warfare on a real service mesh.
> **Status:** Brainstorming

---

## The Concept

We take the distributed system environment from Round 1 and turn it into a competitive adversarial arena:

- **Red Agent (Pen Tester):** Given offensive tools to scan, exploit, and disrupt services — poison messages to Redis, corrupt configs, flood the gateway, plant stale locks.
- **Blue Agent (Sysadmin):** Uses defensive tools to read logs, ban IPs, patch configs, deploy rate limits, restart services, and restore health.

The two agents compete on a **live distributed system** — not a toy grid world, not a text game, but a real service mesh with real Redis, real HTTP routing, real logs.

## Why This Wins

1. **Theme fit:** Multi-agent adversarial interaction with theory-of-mind — Blue must anticipate Red's strategy, Red must adapt to Blue's defenses.
2. **Real professional skill:** Pen testing + incident response are real, high-value cybersecurity skills.
3. **Verifiable reward:** System uptime, throughput, and latency are numbers, not vibes. Perfectly measurable.
4. **R1 leverage:** 60-70% of the infrastructure already exists — mesh services, fault injection, metrics poller, graders, process manager.
5. **Story writes itself:** "We built a live cyber range where two AIs battle — one attacking, one defending. Both get better through self-play."

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CYBERARENA ENVIRONMENT                  │
│                                                             │
│  ┌──────────────┐    ┌─────────────────────────────────┐    │
│  │  RED AGENT   │    │      DISTRIBUTED SYSTEM MESH     │    │
│  │  (Attacker)  │    │                                   │    │
│  │              │───▶│  gateway:3000 ──▶ auth:3001      │    │
│  │ Tools:       │    │       │                           │    │
│  │ • scan       │    │       ▼                           │    │
│  │ • inject     │    │  redis:6379 ──▶ worker ──▶ sqlite │    │
│  │ • flood      │    │                                   │    │
│  │ • tamper     │    └──────────────────┬────────────────┘    │
│  │ • cover      │                       │                    │
│  └──────────────┘                       │ metrics            │
│                                         ▼                    │
│  ┌──────────────┐    ┌─────────────────────────────────┐    │
│  │  BLUE AGENT  │    │        GAME ORCHESTRATOR         │    │
│  │  (Defender)  │    │                                   │    │
│  │              │───▶│  • Turn management                │    │
│  │ Tools:       │    │  • Metrics tracking               │    │
│  │ • read_logs  │    │  • Zero-sum scoring               │    │
│  │ • inspect    │    │  • Episode lifecycle              │    │
│  │ • patch      │    │  • Difficulty curriculum          │    │
│  │ • firewall   │    │                                   │    │
│  │ • restart    │    └─────────────────────────────────┘    │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

### Services (from R1)
- **Gateway** (port 3000): HTTP orchestration, route management
- **Auth** (port 3001): Authentication, configurable delay
- **Redis** (port 6379): Message queue, distributed locks
- **Worker**: Async consumer, job processing
- **SQLite**: Persistence sink

---

## Agent Toolkits

### Red Agent (Offensive)

| Tool | Description | R1 Reuse |
|---|---|---|
| `scan_services` | Discover running services, open ports | New |
| `inject_poison` | Push malformed data to Redis queue | ✅ `inject_byzantine_queue_fault` |
| `tamper_config` | Modify service configs (timeouts, routes, registry) | ✅ fault injector |
| `flood_gateway` | Generate high-volume requests | New |
| `plant_lock` | Create stale distributed locks | ✅ `inject_distributed_lock_starvation` |
| `corrupt_registry` | Modify service registry entries | ✅ `inject_registry_corruption` |
| `block_route` | Add blocked routes in gateway | ✅ `inject_route_partition` |
| `read_logs` | Intelligence gathering from service logs | ✅ exists |
| `cover_tracks` | Clear attack traces from logs | New |

### Blue Agent (Defensive)

| Tool | Description | R1 Reuse |
|---|---|---|
| `read_logs` | Tail/grep service logs | ✅ exists |
| `inspect_redis` | LLEN, LRANGE, KEYS, GET, EXISTS | ✅ exists |
| `inspect_config` | Read service configurations | ✅ exists |
| `patch_config` | Write corrected configs + SIGHUP reload | ✅ exists |
| `restart_service` | Restart a specific service | ✅ exists |
| `clear_lock` | Delete stale Redis locks | ✅ exists |
| `remove_poison` | LREM malformed queue messages | ✅ exists |
| `check_metrics` | Query current system health | ✅ exists |
| `set_rate_limit` | Configure request rate limits on gateway | New |
| `ban_source` | Block traffic from suspicious sources | New |

---

## Reward Design

### Zero-Sum Structure

Every timestep, the system produces measurable metrics. Blue is rewarded for health, Red for damage.

**Blue Team (per step):**
```
R_blue = w1 * throughput_maintained      # requests still being served
       + w2 * uptime_delta               # uptime improved since last step
       + w3 * attack_detected            # detected Red's action
       + w4 * service_recovered          # recovered a downed service
       - w5 * false_positive_penalty     # blocked legitimate traffic
       - w6 * downtime_penalty           # system was down this step
```

**Red Team (per step):**
```
R_red  = w1 * downtime_caused            # system was down
       + w2 * stealth_bonus              # attack not yet detected
       + w3 * cascade_bonus              # attack spread to multiple services
       - w4 * attack_blocked             # Blue reversed the attack
       - w5 * detection_penalty          # Blue identified the vector
```

**Key metrics (all from R1's MetricsPoller):**
- `gateway_success_rate` (float 0-1)
- `gateway_p99_latency_ms` (float)
- `queue_depth` (int)
- `worker_restart_count` (int)
- `consumer_stall_count` (int)

---

## OpenEnv Compatibility

### Challenge
OpenEnv is a single-agent API: `reset() → step(action) → (obs, reward, done)`. We have two agents.

### Solution: Alternating Turns
```
reset(task)  → initial observation (Blue goes first)
step(blue_action)  → obs_blue, reward_blue
step(red_action)   → obs_red, reward_red
step(blue_action)  → obs_blue, reward_blue
...
```

The environment tracks whose turn it is internally. Each `step()` returns the observation and reward for the current agent. This keeps full OpenEnv compatibility.

### Alternative: Two Modes
1. **Training mode (MVP):** Single-agent Blue vs scripted Red. Standard OpenEnv. Red attacks come from a curriculum of scripted attack patterns (R1's fault injector).
2. **Arena mode (demo):** Two-agent adversarial. Extended API or two env instances sharing state.

---

## Training Strategy

### Phase 1: Blue vs Scripted Red (GUARANTEED reward curves)
- Use R1's fault injector as "scripted red team"
- Attack curriculum: easy → medium → hard → multi-vector
  - Easy: single config fault (cascading timeout)
  - Medium: queue poisoning or lock starvation
  - Hard: multi-vector (config corruption + route block simultaneously)
- Train Blue with GRPO via TRL / Unsloth
- **This is essentially R1 reframed as defense training — guaranteed to work**

### Phase 2: Red vs Static Blue (stretch)
- Train Red against rule-based Blue (always checks logs, always restarts)
- Red learns to find creative attack vectors that bypass rote defenses

### Phase 3: Self-Play (demo goal)
- Pit trained Red vs trained Blue
- Show emergent strategies from adversarial co-evolution

> **Scope guard:** Phase 1 alone is a complete, valid submission. Phase 2 makes it strong. Phase 3 makes it legendary. Don't let Phase 3 ambition sabotage Phase 1 delivery.

---

## What's New vs R1

| Component | R1 | CyberArena |
|---|---|---|
| Agent role | Single debugger | Red (attacker) + Blue (defender) |
| Fault source | Scripted injection at reset | Red agent's actions (or scripted curriculum) |
| Goal | Fix a known bug | Ongoing attack/defense over many turns |
| Reward | Progress toward fix | Zero-sum uptime/downtime |
| Difficulty | Fixed per task | Escalating curriculum or adversarial |
| Game dynamics | Single-player puzzle | Competitive multi-agent |

---

## Rough Build Plan

| Task | Effort | Priority |
|---|---|---|
| Port R1 infra to R2 repo, upgrade to OpenEnv latest | 2h | P0 |
| Refactor fault injector into Red agent tool set | 3h | P0 |
| Add new Blue tools (rate limiting, ban source) | 2h | P0 |
| Build game orchestrator (turn management, episode loop) | 2h | P0 |
| Implement zero-sum reward function | 2h | P0 |
| Attack curriculum (scripted Red patterns, easy→hard) | 2h | P0 |
| Training script — Blue vs scripted Red (GRPO/TRL) | 2h | P0 |
| Training run + reward curves | 2h | P0 |
| README + blog/video | 2h | P0 |
| **Total P0** | **~19h** | |
| Red agent training (vs static Blue) | 3h | P1 |
| Self-play adversarial mode | 3h | P2 |
| Live dashboard/visualization | 2h | P2 |

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| Multi-agent training doesn't converge | Fall back to Blue vs scripted Red — still valid for Theme #1 since the environment *supports* multi-agent |
| Game is unbalanced (Red too strong) | Give Red an action budget per episode — N actions, stealth costs more |
| Not enough training time for curves | Short loops showing directional improvement + qualitative before/after |
| OpenEnv API doesn't support two agents cleanly | Use alternating-turn design or single-agent training mode |

---

## The 60-Second Pitch

> "In Round 1, we built an environment where a single agent debugs distributed system failures. But real systems don't fail by accident — they get *attacked*."
>
> "For Round 2, we built **CyberArena**: a live distributed service mesh where two AI agents compete — a pen tester attacking and a sysadmin defending. The attacker learns to find creative exploits. The defender learns to detect, triage, and recover. Through adversarial training, both improve."
>
> "This isn't a toy. It's a real service mesh with real Redis, real HTTP routing, real logs. The same environment a human SRE would work in. And we show that after training, the defender agent reduces mean time to recovery by X% against increasingly sophisticated attack patterns."
