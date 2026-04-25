from typing import Any

from wargames_env.models import WarGamesObservation


def build_red_prompt(
    observation: WarGamesObservation,
    task_name: str,
    step_num: int,
    attempt_history: list[dict[str, Any]],
) -> str:
    history_lines: list[str] = []
    for attempt in attempt_history:
        history_lines.append(
            f"- step {attempt['step']}: command={attempt['command']}; error={attempt['error'] or 'none'}"
        )

    history_block = "\n".join(history_lines) or "- none"
    return (
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
        f"LATEST COMMAND OUTPUT:\n{observation.command_output[:2000]}\n\n"
        'Return exactly one bash command as JSON: {"command":"<bash command>"}.'
    )
