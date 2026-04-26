import argparse
import contextlib
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

sys.path.insert(0, str(Path.cwd()))

from openai import OpenAI

import inference
from wargames_env.models import WarGamesAction
from wargames_env.server import tasks
from wargames_env.server.env import WarGamesEnv
from wargames_env.server.tasks import TaskConfig

TASK_NAME = "phase-2-blue-llm-showdown"
SUMMARY_FIELDS = [
    "task_name",
    "red_model",
    "blue_model",
    "max_steps_cap",
    "actual_steps",
    "success",
    "final_score",
    "max_reward",
    "avg_reward",
    "error",
    "steps_csv",
    "log_path",
]
STEP_FIELDS = [
    "task_name",
    "red_model",
    "blue_model",
    "step",
    "reward",
    "done",
    "termination_reason",
    "red_command",
    "red_reasoning",
    "red_exit_code",
    "red_timed_out",
    "services_affected",
    "services_restored",
    "blue_action_count",
    "blue_kinds",
    "blue_commands",
    "blue_statuses",
    "blue_details",
    "step_error",
    "command_output",
    "raw_blue_actions_json",
]


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def one_line(value: object) -> str:
    return " ".join(str(value).replace("\t", " ").splitlines()).strip()


SERVICE_PATTERNS = {
    "redis": (
        "redis-cli",
        "job_queue",
        "LOCK:job_processor",
        "appendonly",
        "config set save",
    ),
    "gateway": ("gateway", "auth_timeout_ms", "blocked_routes"),
    "auth": ("auth/config", "/auth/", " auth "),
    "worker": (
        "worker/config",
        "db_write_delay_ms",
        "db_pool_size",
        "LOCK:job_processor",
        "/worker/",
        " worker",
    ),
    "job_generator": ("job_generator", "interval_ms"),
}


def _classify_services(text: str) -> str:
    lowered = text.lower()
    matches = [
        service
        for service, patterns in SERVICE_PATTERNS.items()
        if any(pattern.lower() in lowered for pattern in patterns)
    ]
    return ",".join(matches)


def classify_services_affected(red_command: str) -> str:
    return _classify_services(red_command)


def classify_services_restored(blue_actions: list[dict[str, Any]]) -> str:
    text = " ".join(
        one_line(action.get("target", "")) + " " + one_line(action.get("detail", ""))
        for action in blue_actions
    )
    return _classify_services(text)


def folder_label(model: str) -> str:
    raw = model.split("/", 1)[-1].lower()
    raw = raw.replace("claude-", "claude_")
    raw = raw.replace("gpt-", "gpt")
    raw = raw.replace("llama-", "llama")
    raw = raw.replace("qwen", "qwen")
    raw = raw.replace(".", "")
    raw = raw.replace("-", "_")
    raw = re.sub(r"[^a-z0-9_]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    raw = re.sub(r"_([0-9]+)$", r"\1", raw)
    return raw


def parse_models(value: str) -> list[str]:
    return [model.strip() for model in value.split(",") if model.strip()]


def _apply_common_env(model: str, max_steps: int) -> None:
    os.environ["MODEL_NAME"] = model
    os.environ["BLUE_MODEL_NAME"] = model
    os.environ["MAX_COMPLETION_TOKENS"] = os.getenv("MAX_COMPLETION_TOKENS", "2048")

    tasks.TASK_CONFIGS[TASK_NAME] = TaskConfig(TASK_NAME, max_steps)
    inference.MODEL_NAME = model
    inference.MAX_STEPS_CAP = max_steps
    inference.MAX_COMPLETION_TOKENS = int(os.environ["MAX_COMPLETION_TOKENS"])


def configure_openrouter(model: str, max_steps: int) -> None:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY must be set")

    os.environ["API_KEY"] = key
    os.environ["BLUE_API_KEY"] = key
    os.environ["API_BASE_URL"] = "https://openrouter.ai/api/v1"
    os.environ["BLUE_API_BASE_URL"] = "https://openrouter.ai/api/v1"
    os.environ["CHAT_TOKEN_LIMIT_PARAM"] = "max_tokens"

    inference.API_BASE_URL = os.environ["API_BASE_URL"]
    inference.API_KEY = os.environ["API_KEY"]

    _apply_common_env(model, max_steps)


def configure_hf(model: str, max_steps: int) -> None:
    key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    if not key:
        raise RuntimeError("HF_TOKEN (or API_KEY) must be set")

    os.environ["API_KEY"] = key
    os.environ["BLUE_API_KEY"] = key
    os.environ["API_BASE_URL"] = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    os.environ["BLUE_API_BASE_URL"] = os.getenv("BLUE_API_BASE_URL", os.environ["API_BASE_URL"])
    os.environ["CHAT_TOKEN_LIMIT_PARAM"] = os.getenv("CHAT_TOKEN_LIMIT_PARAM", "max_tokens")

    inference.API_BASE_URL = os.environ["API_BASE_URL"]
    inference.API_KEY = os.environ["API_KEY"]

    _apply_common_env(model, max_steps)


def reset_redis() -> None:
    subprocess.run(["mkdir", "-p", "/tmp"], check=False)
    subprocess.run(
        ["redis-cli", "shutdown", "nosave"], check=False, capture_output=True, text=True
    )
    subprocess.run(
        [
            "redis-server",
            "--daemonize",
            "yes",
            "--logfile",
            "/tmp/redis.log",
            "--port",
            "6379",
        ],
        check=False,
    )
    subprocess.run(["redis-cli", "ping"], check=False, capture_output=True, text=True)


def episode_score(rewards: list[float]) -> float:
    if not rewards:
        return 0.01
    return max(0.01, min(0.99, max(float(reward) for reward in rewards)))


def flatten_blue_actions(actions: list[dict[str, Any]]) -> tuple[str, str, str, str]:
    kinds: list[str] = []
    commands: list[str] = []
    statuses: list[str] = []
    details: list[str] = []
    for action in actions:
        kinds.append(one_line(action.get("kind", "")))
        commands.append(one_line(action.get("target", "")))
        statuses.append(one_line(action.get("status", "")))
        details.append(one_line(action.get("detail", "")))
    return (
        " || ".join(kinds),
        " || ".join(commands),
        " || ".join(statuses),
        " || ".join(details),
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_model(model: str, max_steps: int, output_root: Path, timestamp: str, *, provider: str = "openrouter") -> Path:
    configure = configure_openrouter if provider == "openrouter" else configure_hf
    output_dir = output_root / f"docker_{provider}_{folder_label(model)}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    steps_path = output_dir / "steps.csv"
    log_path = output_dir / "red_vs_blue.log"

    configure(model, max_steps)
    reset_redis()

    client = OpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.environ["API_BASE_URL"],
        timeout=30.0,
        max_retries=1,
    )
    env = WarGamesEnv(project_root=Path.cwd(), mesh_root=Path.cwd() / "mesh")
    messages = [{"role": "system", "content": inference.SYSTEM_PROMPT}]
    attempt_history: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    rewards: list[float] = []
    error = ""
    done = False
    step = 0

    with (
        log_path.open("w", encoding="utf-8") as log_file,
        contextlib.redirect_stdout(Tee(sys.stdout, log_file)),
    ):
        print(
            f"[START] task={TASK_NAME} env=docker red_model={model} blue_model={model} max_steps={max_steps}",
            flush=True,
        )
        try:
            obs = env.reset(task_name=TASK_NAME)
            state = SimpleNamespace(**env.state())
            episode_steps = min(state.max_steps, max_steps)
            while not done and step < episode_steps:
                next_step = step + 1
                messages.append(
                    {
                        "role": "user",
                        "content": inference.build_prompt(
                            obs, next_step, TASK_NAME, attempt_history
                        ),
                    }
                )
                completion = client.chat.completions.create(
                    model=model,
                    messages=cast(Any, messages),
                    temperature=inference.TEMPERATURE,
                    **cast(Any, inference._chat_token_limit_kwargs()),
                )
                raw_response = inference._assistant_message_text(
                    completion.choices[0].message
                )
                red_command, red_reasoning = inference.extract_action_payload(
                    raw_response
                )
                if not red_command:
                    messages.append({"role": "assistant", "content": raw_response})
                    red_command = inference.NO_COMMAND_PROVIDED_SENTINEL
                    red_reasoning = None
                else:
                    assistant_payload = {"command": red_command}
                    if red_reasoning:
                        assistant_payload["reasoning"] = red_reasoning
                    messages.append(
                        {"role": "assistant", "content": json.dumps(assistant_payload)}
                    )

                result = env.step(
                    WarGamesAction(command=red_command, reasoning=red_reasoning)
                )
                obs = result.observation
                rewards.append(result.reward)
                done = result.done
                info = result.info
                blue_actions = (
                    info.get("blue_actions", []) if isinstance(info, dict) else []
                )
                if not isinstance(blue_actions, list):
                    blue_actions = []
                blue_kinds, blue_commands, blue_statuses, blue_details = (
                    flatten_blue_actions(blue_actions)
                )
                error_value = info.get("error") if isinstance(info, dict) else None
                step_error = (
                    "" if error_value in (None, "", "None") else one_line(error_value)
                )
                row = {
                    "task_name": TASK_NAME,
                    "red_model": model,
                    "blue_model": model,
                    "step": next_step,
                    "reward": f"{result.reward:.4f}",
                    "done": str(done).lower(),
                    "termination_reason": (
                        info.get("termination_reason", "")
                        if isinstance(info, dict)
                        else ""
                    ),
                    "red_command": one_line(red_command),
                    "red_reasoning": one_line(red_reasoning or ""),
                    "red_exit_code": (
                        info.get("exit_code", "") if isinstance(info, dict) else ""
                    ),
                    "red_timed_out": (
                        info.get("timed_out", "") if isinstance(info, dict) else ""
                    ),
                    "services_affected": classify_services_affected(red_command),
                    "services_restored": classify_services_restored(blue_actions),
                    "blue_action_count": len(blue_actions),
                    "blue_kinds": blue_kinds,
                    "blue_commands": blue_commands,
                    "blue_statuses": blue_statuses,
                    "blue_details": blue_details,
                    "step_error": step_error,
                    "command_output": one_line(obs.command_output),
                    "raw_blue_actions_json": json.dumps(blue_actions, sort_keys=True),
                }
                step_rows.append(row)
                attempt_history.append(
                    {
                        "step": next_step,
                        "command": red_command,
                        "reasoning": red_reasoning,
                        "output": obs.command_output,
                        "error": step_error or None,
                    }
                )
                print(
                    f"[STEP] step={next_step} reward={result.reward:.2f} done={str(done).lower()} "
                    f"red={one_line(red_command)} red_reasoning={one_line(red_reasoning or '')} "
                    f"blue_count={len(blue_actions)} blue_status={blue_statuses or 'none'} "
                    f"blue_command={blue_commands or 'none'} blue_detail={blue_details or 'none'} "
                    f"output={one_line(obs.command_output)[:500]} error={step_error or 'null'}",
                    flush=True,
                )
                step = next_step
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] task={TASK_NAME} {error}", flush=True)
        finally:
            final_score = episode_score(rewards)
            max_reward = max(rewards) if rewards else 0.0
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            success = bool(done and final_score >= 0.95)
            print(
                f"[END] success={str(success).lower()} steps={step} final_score={final_score:.2f} "
                f"max_reward={max_reward:.2f} avg_reward={avg_reward:.2f} error={one_line(error) or 'null'}",
                flush=True,
            )
            env.close()
            client.close()

    summary_rows = [
        {
            "task_name": TASK_NAME,
            "red_model": model,
            "blue_model": model,
            "max_steps_cap": max_steps,
            "actual_steps": step,
            "success": str(bool(done and episode_score(rewards) >= 0.95)).lower(),
            "final_score": f"{episode_score(rewards):.4f}",
            "max_reward": f"{(max(rewards) if rewards else 0.0):.4f}",
            "avg_reward": f"{(sum(rewards) / len(rewards) if rewards else 0.0):.4f}",
            "error": error,
            "steps_csv": str(steps_path),
            "log_path": str(log_path),
        }
    ]
    write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    write_csv(steps_path, STEP_FIELDS, step_rows)
    print(f"[SUMMARY_CSV] {summary_path}", flush=True)
    print(f"[STEPS_CSV] {steps_path}", flush=True)
    print(f"[LOG] {log_path}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--provider", choices=["openrouter", "hf"], default="openrouter", help="LLM provider (default: openrouter)")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    for model in parse_models(args.models):
        run_model(
            model=model,
            max_steps=args.max_steps,
            output_root=output_root,
            timestamp=args.timestamp,
            provider=args.provider,
        )


if __name__ == "__main__":
    main()
