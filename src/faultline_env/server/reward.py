from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from faultline_env.models import SystemMetrics

POSITIVE_WEIGHT_TOTAL = 3.0


@dataclass(frozen=True)
class RewardContext:
    metrics_before: SystemMetrics
    metrics_after_red: SystemMetrics
    metrics_after_blue: SystemMetrics
    command: str
    recent_commands: list[str] = field(default_factory=list)
    process_status_before: dict[str, str] | None = None
    process_status_after_red: dict[str, str] | None = None
    process_status_after_blue: dict[str, str] | None = None


class RewardBreakdown(BaseModel):
    total: float = Field(..., ge=0.0, le=1.0)
    components: dict[str, float]
    weighted_components: dict[str, float]


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _positive_damage_score(before: SystemMetrics, after: SystemMetrics) -> float:
    success_rate_drop = _clip(before.gateway_success_rate - after.gateway_success_rate)
    latency_spike = _clip(
        (after.gateway_p99_latency_ms - before.gateway_p99_latency_ms) / 1000.0
    )
    queue_backup = _clip((after.queue_depth - before.queue_depth) / 25.0)
    return (2.0 * success_rate_drop) + (0.5 * latency_spike) + (0.3 * queue_backup)


def _stopped_services(
    before: dict[str, str] | None, after: dict[str, str] | None
) -> set[str]:
    if not before or not after:
        return set()
    stopped = set()
    for service, before_status in before.items():
        after_status = after.get(service, "")
        if "running" in before_status and "running" not in after_status:
            stopped.add(service)
    return stopped


def _command_contains_any(command: str, fragments: tuple[str, ...]) -> bool:
    lowered = command.lower()
    return any(fragment in lowered for fragment in fragments)


def compute_red_reward(context: RewardContext) -> RewardBreakdown:
    stopped_after_red = _stopped_services(
        context.process_status_before, context.process_status_after_red
    )
    stopped_after_blue = _stopped_services(
        context.process_status_before, context.process_status_after_blue
    )
    critical_services = {"gateway", "auth", "worker"}
    config_tamper = _command_contains_any(
        context.command,
        (
            "> /mesh/",
            "> /home/user/app/mesh/",
            "config.json",
            "job_generator_config.json",
            "blocked_routes.json",
            "chmod ",
        ),
    )
    redis_attack = _command_contains_any(
        context.command,
        (
            "redis-cli set lock:job_processor",
            "redis-cli del job_queue",
            "redis-cli flush",
            "redis-cli config set",
            "redis-cli rename",
            "redis-cli rpush job_queue",
            "redis-cli lpush job_queue",
        ),
    )
    components = {
        "success_rate_drop": _clip(
            context.metrics_before.gateway_success_rate
            - context.metrics_after_red.gateway_success_rate
        ),
        "latency_spike": _clip(
            (
                context.metrics_after_red.gateway_p99_latency_ms
                - context.metrics_before.gateway_p99_latency_ms
            )
            / 1000.0
        ),
        "queue_backup": _clip(
            (context.metrics_after_red.queue_depth - context.metrics_before.queue_depth)
            / 25.0
        ),
        "stealth_bonus": 0.0,
        "no_op_penalty": 0.0,
        "repeat_penalty": 0.0,
        "service_down": _clip(len(stopped_after_red) / 4.0),
        "critical_service_down": _clip(
            len(stopped_after_red & critical_services) / len(critical_services)
        ),
        "persistent_service_down": _clip(len(stopped_after_blue) / 4.0),
        "config_tamper": 1.0 if config_tamper else 0.0,
        "redis_attack": 1.0 if redis_attack else 0.0,
    }

    after_red_damage = _positive_damage_score(
        context.metrics_before, context.metrics_after_red
    )
    after_blue_damage = _positive_damage_score(
        context.metrics_before, context.metrics_after_blue
    )
    if after_red_damage > 0 and after_blue_damage > 0:
        components["stealth_bonus"] = _clip(after_blue_damage / after_red_damage)

    direct_damage = (
        components["service_down"]
        + components["critical_service_down"]
        + components["config_tamper"]
        + components["redis_attack"]
    )
    if after_red_damage == 0 and direct_damage == 0:
        components["no_op_penalty"] = 1.0

    if context.command.strip() in [
        command.strip() for command in context.recent_commands
    ]:
        components["repeat_penalty"] = 1.0

    weighted_components = {
        "success_rate_drop": 2.0 * components["success_rate_drop"],
        "latency_spike": 0.5 * components["latency_spike"],
        "queue_backup": 0.3 * components["queue_backup"],
        "stealth_bonus": 0.2 * components["stealth_bonus"],
        "service_down": 0.8 * components["service_down"],
        "critical_service_down": 1.2 * components["critical_service_down"],
        "persistent_service_down": 0.6 * components["persistent_service_down"],
        "config_tamper": 0.25 * components["config_tamper"],
        "redis_attack": 0.25 * components["redis_attack"],
        "no_op_penalty": -0.1 * components["no_op_penalty"],
        "repeat_penalty": -0.1 * components["repeat_penalty"],
    }
    total = _clip(sum(weighted_components.values()) / POSITIVE_WEIGHT_TOTAL)
    return RewardBreakdown(
        total=total,
        components=components,
        weighted_components=weighted_components,
    )
