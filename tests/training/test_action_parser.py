from training.env_adapter.action_parser import (
    ActionParseError,
    parse_model_command,
    parse_model_command_strict,
)

import pytest


def test_parse_model_command_reads_json_and_falls_back_to_single_line():
    assert (
        parse_model_command('{"command":"redis-cli LLEN job_queue"}')
        == "redis-cli LLEN job_queue"
    )
    assert parse_model_command("curl localhost:3000/health") == "curl localhost:3000/health"


def test_strict_parser_accepts_qwen3_thinking_block():
    completion = (
        "<think>\nReasoning about which command to send.\n</think>\n\n"
        '{"command": "redis-cli LLEN job_queue"}'
    )
    assert parse_model_command_strict(completion) == "redis-cli LLEN job_queue"


def test_strict_parser_accepts_bare_json():
    assert (
        parse_model_command_strict('{"command": "ls -la"}')
        == "ls -la"
    )


def test_strict_parser_rejects_truncated_thinking():
    with pytest.raises(ActionParseError):
        parse_model_command_strict("<think>\nstill thinking, ran out of tokens")


def test_strict_parser_rejects_prose_after_thinking():
    with pytest.raises(ActionParseError):
        parse_model_command_strict("<think>\nthinking\n</think>\n\nI will run ls -la")


def test_strict_parser_rejects_empty_command_field():
    with pytest.raises(ActionParseError):
        parse_model_command_strict('{"command": "   "}')


def test_strict_parser_rejects_missing_command_field():
    with pytest.raises(ActionParseError):
        parse_model_command_strict('{"cmd": "ls"}')


def test_strict_parser_rejects_empty_completion():
    with pytest.raises(ActionParseError):
        parse_model_command_strict("")
    with pytest.raises(ActionParseError):
        parse_model_command_strict("<think>\nthinking\n</think>\n\n")