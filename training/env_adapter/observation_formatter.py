"""Format env observations into the Red agent prompt.

Includes the Red system prompt, the task name, current metrics, process status,
the prior attempt history (with truncation note), and the latest command output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from faultline_env.models import FaultlineObservation

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "red_system_prompt.txt"
_COMMAND_OUTPUT_LIMIT = 2000


def _load_system_prompt() -> str:
    try:
        return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return (
            "You are the Red Team agent attacking a live distributed job processing "
            "pipeline. Return exactly one bash command as JSON with a required "
            "`command` field."
        )


_SYSTEM_PROMPT = _load_system_prompt()


def build_red_prompt(
    observation: FaultlineObservation,
    task_name: str,
    step_num: int,
    attempt_history: list[dict[str, Any]],
) -> str:
    history_lines: list[str] = []
    for attempt in attempt_history:
        history_lines.append(
            f"- step {attempt['step']}: command={attempt['command']}; "
            f"error={attempt['error'] or 'none'}"
        )
    history_block = "\n".join(history_lines) or "- none"

    raw_output = observation.command_output
    truncated = len(raw_output) > _COMMAND_OUTPUT_LIMIT
    output_block = raw_output[:_COMMAND_OUTPUT_LIMIT]
    if truncated:
        output_block += f"\n[...truncated to {_COMMAND_OUTPUT_LIMIT} chars]"

    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Step {step_num}.\n\n"
        f"TASK: {task_name}\n"
        "METRICS:\n"
        f"- Gateway success rate: {observation.metrics.gateway_success_rate:.1%}\n"
        f"- Gateway P99 latency: {observation.metrics.gateway_p99_latency_ms:.0f}ms\n"
        f"- Queue depth: {observation.metrics.queue_depth}\n"
        f"- Worker restarts: {observation.metrics.worker_restart_count}\n"
        f"- Consumer stall count: {observation.metrics.consumer_stall_count}\n\n"
        f"SERVICE STATUS:\n{observation.process_status}\n\n"
        f"PREVIOUS ATTEMPTS:\n{history_block}\n\n"
        f"LATEST COMMAND OUTPUT:\n{output_block}\n\n"
        'Return exactly one bash command as JSON: {"command":"<bash command>"}.'
    )
