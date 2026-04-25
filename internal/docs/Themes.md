# Research Themes

## Theme #1: Multi-Agent Interactions
Environments for this theme involve cooperation, competition, negotiation, and coalition formation. Learning from these environments will enable agents to model the beliefs and incentives of others in partially observable settings. This drives theory-of-mind reasoning and emergent strategic behavior.

*   **Expected Outcome:** An environment that can be used to train multi-agent task handling in an LLM.
*   **Example Environments:** Market simulations, compute-allocation negotiations, collaborative puzzle worlds, mixed cooperative/competitive strategy games.

---

## Theme #2: (Super) Long-Horizon Planning & Instruction Following
You will build environments that require deep, multi-step reasoning with sparse or delayed rewards. After using these environments, the goal is to enable agents to decompose goals, track state over extended trajectories, and recover from early mistakes. The aim is to push beyond shallow next-token reasoning toward structured planning and durable internal representations.

*   **Expected Outcome:** An environment that can capture and improve LLM behaviour on challenging long-horizon tasks that need long-running sessions beyond context memory limits.
*   **Example Environments:** OpenClaw workflows with multi-turn tasks, research-planning simulators, large-scale codebase refactoring tasks, strategic resource management worlds, long-horizon logistics optimization, extremely complicated long-horizon instruction following (e.g., 300 instructions scattered around).

---

## Theme #3: World Modeling

### #3.1 Professional Tasks
Here you will develop environments that require real interaction with tools, APIs, or dynamic systems where the model is expected to do real hard work instead of exploiting shortcuts to arrive at the desired outcome. Learning from these environments will enable agents to maintain consistent internal state, update beliefs based on outcomes, and orchestrate multi-step workflows. The goal is to strengthen causal reasoning and persistent world models.

*   **Expected Outcome:** An environment capturing nuances of a defined partially observable world and improve LLM interaction with it.
*   **Example Environments:** Dynamic browser/API ecosystems, enterprise applications, scientific workflow loops (papers → code → experiments), economic simulations with feedback, tool-discovery benchmarks.

### #3.2 Personalized Tasks
Here we will develop an environment that offers real personalized task handling—imagine replying to personal messages, handling scheduling conflicts, or drafting difficult emails. Think of any personal assistant task.

*   **Expected Outcome:** An environment that gives the model a realistic simulation of handling personal tasks, managing conflicts, and handling delegations.
*   **Example Environments:** Executive Assistant meeting planner, dinner and drive planning, email/message replying, shopping, etc.

---

## Theme #4: Self-Improvement
The focus here is to create environments where agents can learn to generate new challenges, escalate difficulty, and improve through self-play or adaptive curricula. Rather than optimizing fixed tasks, the goal is for agents to learn to drive their own capability growth. The objective is recursive skill amplification.

*   **Expected Outcome:** An environment for improving self-play of an LLM over a defined set of tasks.
*   **Example Environments:** Self-play negotiation arenas, auto-generated math/proof tasks, evolving coding competitions, adaptive RL curricula.

---

## Theme #5: Wild Card - Impress Us!
We do not want to limit your focus if your idea doesn’t fit the boxes above. We want and will reward "out of the box" tasks. Please be creative, but remember to add submissions that meaningfully add value to LLM training on a certain task.

## Guidelines for Problem Statement
It is NOT mandatory to choose the same problem statement as Round 1. Only choose the same problem statement if it aligns with the above provided Hackathon themes.
You can start working on your problem statement once you have finalized it. Post-training can be done onsite on 25th & 26th when you receive compute credits for HuggingFace.
Before the onsite, we suggest you work on building the environment, agent behaviours, reward model and evaluate if your work aligns with the judging criteria given below.
