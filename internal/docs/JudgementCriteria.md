# Judging Criteria

## Minimum Requirements
*   Usage of OpenEnv (latest release).
*   Show a minimal training script for your environment using Unsloth or HF TRL in Colab.
*   Write a mini-blog on HuggingFace or mini-video on YouTube talking about your submission, < 2 minutes.
*   Your OpenEnv compliant environment should be hosted on Hugging Face Spaces.

## Judging Overview
Evaluation: Teams will be scored based on the following criteria:

*   **Environment Innovation (40%):** Is the environment novel, creative, or challenging? Does it meaningfully test the agent’s behavior?
*   **Storytelling (30%):** Does the team clearly explain the problem, environment, and agent behavior? Is the demo engaging and easy to follow?
*   **Showing Improvement in Rewards (20%):** Does the demo provide observable evidence of training progress (reward curves, metrics, or before/after behavior)?
*   **Reward and Training Script/Pipeline Setup (10%):** Is the reward logic coherent, and does the pipeline produce meaningful improvement in the agent’s inference (how it acts in the environment)?

---

## OpenEnv Hackathon - What Judges Look For
This guide tells you what makes a strong submission for the OpenEnv Hackathon (India 2026). Read it before you start building, and again before you submit.

For the list of themes and example problems, refer to the top sections.

**NOTE:** Please remember only one submission per team. If you have multiple ideas, pick the best one and go for it. Please make sure that the URL link of your environment is submitted as judges will pull the environment from the URL to evaluate it. Changes or commits after the submission deadline will not be considered.

### TL;DR
Build an environment that an LLM could actually be trained on to get measurably better at something interesting. Then show that training. Then tell the story.

A messy but ambitious environment with real training evidence beats a polished but boring one. Pick a problem that excites you (that energy comes through in the pitch).

### Criterion: Environment Innovation (40%)
*   **What it means:**
    *   Is the environment novel, creative, or genuinely challenging?
    *   Does it meaningfully test agent behavior in a way that hasn't been done before?

### Criterion: Storytelling & Presentation (30%)
*   **What it means:**
    *   Can you clearly explain the problem, the environment, and what the agent learned?
    *   Is the demo engaging and easy to follow for a non-technical audience?

### Criterion: Showing Improvement in Rewards (20%)
*   **What it means:**
    *   Is there observable evidence of training progress? Reward curves, before/after behavior, comparison against a baseline -- anything that proves the agent learned something.

### Criterion: Reward & Training Pipeline (10%)
*   **What it means:**
    *   Is the reward logic coherent? Does the pipeline produce meaningful improvement in the trained agent's behavior?

---

## What Makes a Submission Stand Out

### Pick an ambitious, original problem
The themes (problems) are deliberately open. Use them as launching pads, not boxes. Judges have seen a lot of chess, snake, tic-tac-toe, and grid-world clones. To score well on innovation, you need a genuinely fresh angle. Some questions to ask yourself:
*   Does this environment exist to teach an LLM something it currently can’t do well?
*   Is the domain underexplored in RL/LLM training?
*   Could a researcher write a paper about training on this?

### Design a reward signal that actually teaches
A great environment has a reward function that:
*   Provides a rich, informative signal (not just 0/1 at the end).
*   Captures something hard to measure in a clever way.
*   Uses OpenEnv’s Rubric system thoughtfully (composable rubrics > monolithic scoring).
*   Is hard to game; an agent that exploits the reward without solving the task should not get high scores.

### Show real training, end to end
The bar isn’t “training script exists.” The bar is “training script runs against the environment, the agent learns, and you can show it.” Concretely:
*   Your training loop should connect to your environment (not a static dataset).
*   Train long enough that the curves mean something.
*   Compare a trained agent vs. a random/untrained baseline; quantitative and/or qualitative.
*   Include the plots and numbers in your README and writeup.

### Make your plots readable
Reviewers spend seconds, not minutes, on each plot. Help them out:
*   Label both axes (e.g., “training step” / “episode” on x, “reward” / “loss” on y) and include units where they apply.
*   Save plots as .png or .jpg and commit them to the repo (don’t leave them only in a Colab cell or a deleted Wandb run) (if you ran via Wandb, please include the link to that specific run of your plots).
*   Embed the key plots in your README with a one-line caption explaining what each one shows. If you have multiple runs (baseline vs. trained, ablations, etc.), put them on the same axes so the comparison is obvious.

### Tell a story, not an API doc
Your README, blog, and pitch should answer:
*   **Problem:** What capability gap or interesting domain are you targeting?
*   **Environment:** What does the agent see, do, and get rewarded for?
*   **Results:** What changed after training? Show it.
*   **Why does it matter:** Who would care, and why?

A reviewer should be able to read your README in 3~5 minutes and want to try your environment.

**NOTE:** If you have a video, HF post, or anything else interesting, please make sure that it’s linked from your README as a link.

### Engineer it cleanly (table stakes)
Engineering quality matters less than ambition, but sloppy work hurts. Make sure you:
*   Use OpenEnv’s `Environment` / `MCPEnvironment` base classes properly.
*   Respect the client / server separation (clients should never import server internals).
*   Follow the standard Gym-style API (reset, step, state).
*   Have a valid `openenv.yaml` manifest.
*   Don’t use reserved tool names (reset, step, state, close) for MCP tools.

## Final Note
Judges are looking for environments that push the frontier of what we can train LLMs to do. Be ambitious. Pick a problem you find genuinely interesting; that almost always produces better work than chasing what you think judges want. Good luck.
