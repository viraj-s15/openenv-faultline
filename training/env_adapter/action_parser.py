"""Action parsing for the Red agent.

Two modes:

- `parse_model_command` (lenient): used by inference where messy upstream
  models routinely return prose. Falls back to the first non-empty line.
- `parse_model_command_strict`: used by training to refuse ambiguous output
  so the reward signal directly reflects format compliance.
"""

from __future__ import annotations

import json

NO_COMMAND_PROVIDED = "echo NO_COMMAND_PROVIDED"


class ActionParseError(ValueError):
    """Raised by the strict parser when input does not satisfy the JSON contract."""


def parse_model_command(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return NO_COMMAND_PROVIDED

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        first_line = stripped.splitlines()[0].strip()
        return first_line or NO_COMMAND_PROVIDED

    if isinstance(payload, dict) and isinstance(payload.get("command"), str):
        command = payload["command"].strip()
        return command or NO_COMMAND_PROVIDED

    return NO_COMMAND_PROVIDED


def parse_model_command_strict(text: str) -> str:
    """Strict variant: requires `{"command": "<bash>"}` JSON, no prose, no fallback.

    Returns the bash command. Raises `ActionParseError` on any deviation so the
    caller (training rollout) can record the failure as a real signal.
    """
    stripped = text.strip()
    if not stripped:
        raise ActionParseError("empty completion")

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ActionParseError(f"completion is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ActionParseError("completion JSON must be an object")

    command = payload.get("command")
    if not isinstance(command, str):
        raise ActionParseError("completion JSON missing string `command` field")

    command = command.strip()
    if not command:
        raise ActionParseError("`command` field is empty")
    return command
