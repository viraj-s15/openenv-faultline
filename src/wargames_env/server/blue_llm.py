import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from wargames_env.models import SystemMetrics

BLUE_SYSTEM_PROMPT = """You are the incident commander for a live distributed system.
Keep metrics green while a Red agent attacks the mesh.
You have bash access with standard tools: ps, ls, cat, tail, curl, jq, redis-cli, kill, sed, netstat, ss, lsof, ping, dig.
Prefer mesh-native defensive actions: inspect logs, restore configs, send SIGHUP, restart services, sanitize Redis.
Respond with compact JSON where `command` is required: {"command":"<bash command>","reasoning":"optional concise reason"}.
No markdown. No explanation outside JSON."""

Provider = Callable[[list[dict[str, str]]], str]
_JSON_DECODER = json.JSONDecoder()


@dataclass(frozen=True)
class BlueLLMCommandResult:
    command: str
    status: str
    detail: str


def _single_line(text: str) -> str:
    return " ".join(text.replace("\t", " ").splitlines()).strip()


def _command_from_dict(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    command_value = payload.get("command")
    command = command_value.strip() if isinstance(command_value, str) else ""
    if not command:
        return None, None

    reasoning_value = payload.get("reasoning")
    reasoning = reasoning_value.strip() if isinstance(reasoning_value, str) else ""
    return command, reasoning or None


def _parse_action_payload(text: str) -> tuple[str | None, str | None]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None, None

    if not isinstance(payload, dict):
        return None, None
    return _command_from_dict(payload)


def _iter_decoded_json_objects(text: str) -> Iterator[Any]:
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = _JSON_DECODER.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        yield obj
        i = end


def extract_action_payload(llm_response: str) -> tuple[str | None, str | None]:
    response = llm_response.strip()
    if not response:
        return None, None

    if response.startswith("```"):
        lines = response.split("\n")
        if len(lines) > 2:
            response = "\n".join(lines[1:-1]).strip()

    direct_command, direct_reasoning = _parse_action_payload(response)
    if direct_command:
        return direct_command, direct_reasoning

    for obj in _iter_decoded_json_objects(response):
        if isinstance(obj, dict):
            embedded_command, embedded_reasoning = _command_from_dict(obj)
            if embedded_command:
                return embedded_command, embedded_reasoning

    for match in re.finditer(r"\{[^{}]*\}", response, flags=re.DOTALL):
        embedded_command, embedded_reasoning = _parse_action_payload(match.group(0))
        if embedded_command:
            return embedded_command, embedded_reasoning

    first_line = response.split("\n")[0].strip()
    return _parse_action_payload(first_line)


def build_blue_prompt(
    metrics: SystemMetrics,
    process_status: dict[str, str],
    red_command: str,
    red_reasoning: object | None = None,
    project_root: Path | None = None,
    mesh_root: Path | None = None,
) -> str:
    root = project_root or Path(os.getenv("APP_ROOT", "/home/user/app"))
    mesh = mesh_root or Path(os.getenv("MESH_ROOT", "/mesh"))
    reasoning = _single_line(str(red_reasoning or "")) or "(none provided)"
    return (
        "Current system state:\n\n"
        "LATEST RED COMMAND:\n"
        f"{red_command}\n\n"
        "RED REASONING:\n"
        f"{reasoning}\n\n"
        "METRICS:\n"
        f"- Gateway success rate: {metrics.gateway_success_rate:.1%}\n"
        f"- Gateway P99 latency: {metrics.gateway_p99_latency_ms:.0f}ms\n"
        f"- Queue depth: {metrics.queue_depth}\n"
        f"- Worker restarts: {metrics.worker_restart_count}\n"
        f"- Consumer stall count: {metrics.consumer_stall_count}\n\n"
        "SERVICE STATUS:\n"
        f"{process_status}\n\n"
        "RUNTIME CONTRACT:\n"
        f"- App root: {root}\n"
        f"- Mesh root: {mesh}\n"
        "- Redis queue key: job_queue\n"
        "- Logs: /tmp/gateway.log, /tmp/auth.log, /tmp/worker.log, /tmp/job_gen.log\n"
        "- Configs: gateway/blocked_routes.json, gateway/config.json, auth/config.json, worker/config.json, worker/job_generator_config.json\n"
        "- Goal: keep services healthy, restore tampered configs, and sanitize poisoned Redis state.\n\n"
        "Return the single next defensive bash command."
    )


def build_default_blue_provider() -> Provider | None:
    api_key = os.getenv("BLUE_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    if not api_key:
        return None

    api_base_url = os.getenv("BLUE_API_BASE_URL") or os.getenv(
        "API_BASE_URL", "https://router.huggingface.co/v1"
    )
    model_name = os.getenv("BLUE_MODEL_NAME") or os.getenv(
        "MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"
    )
    temperature = float(os.getenv("BLUE_TEMPERATURE", os.getenv("TEMPERATURE", "0.2")))
    max_completion_tokens = int(
        os.getenv(
            "BLUE_MAX_COMPLETION_TOKENS",
            os.getenv("MAX_COMPLETION_TOKENS", "2048"),
        )
    )
    token_param = os.getenv("CHAT_TOKEN_LIMIT_PARAM", "").strip().lower()

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=api_base_url, timeout=30.0, max_retries=2)

    def provider(messages: list[dict[str, str]]) -> str:
        token_kwargs = (
            {"max_completion_tokens": max_completion_tokens}
            if token_param == "max_completion_tokens"
            or "api.openai.com" in api_base_url
            else {"max_tokens": max_completion_tokens}
        )
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            **token_kwargs,
        )
        return str(completion.choices[0].message.content or "")

    return provider


def run_blue_llm_tick(
    provider: Provider,
    process_manager: object,
    metrics_poller: object,
    project_root: Path,
    mesh_root: Path,
    red_command: str,
    red_reasoning: object | None = None,
) -> BlueLLMCommandResult:
    metrics_poller.poll_once()
    metrics = metrics_poller.get_current_metrics()
    process_status = process_manager.get_status()
    messages = [
        {"role": "system", "content": BLUE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_blue_prompt(
                metrics=metrics,
                process_status=process_status,
                red_command=red_command,
                red_reasoning=red_reasoning,
                project_root=project_root,
                mesh_root=mesh_root,
            ),
        },
    ]
    raw_response = provider(messages)
    command, reasoning = extract_action_payload(raw_response)
    if not command:
        return BlueLLMCommandResult(
            command="",
            status="error",
            detail="no defensive command provided",
        )

    shell_path = ":".join(
        path
        for path in [
            os.environ.get("PATH", ""),
            "/usr/local/sbin",
            "/usr/local/bin",
            "/usr/sbin",
            "/usr/bin",
            "/sbin",
            "/bin",
        ]
        if path
    )
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=10,
        cwd="/",
        env={
            **os.environ,
            "APP_ROOT": str(project_root),
            "MESH_ROOT": str(mesh_root),
            "PATH": shell_path,
        },
        check=False,
    )
    output = _single_line((result.stdout + result.stderr).strip() or "(no output)")
    detail_parts = [f"exit_code={result.returncode}"]
    if reasoning:
        detail_parts.append(f"reasoning={_single_line(reasoning)[:160]}")
    detail_parts.append(f"output={output[:160]}")
    return BlueLLMCommandResult(
        command=command,
        status="applied" if result.returncode == 0 else "error",
        detail=" ".join(detail_parts),
    )
