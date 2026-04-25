import json


NO_COMMAND_PROVIDED = "echo NO_COMMAND_PROVIDED"


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
