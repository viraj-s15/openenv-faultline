from training.env_adapter.action_parser import parse_model_command


def test_parse_model_command_reads_json_and_falls_back_to_single_line():
    assert (
        parse_model_command('{"command":"redis-cli LLEN job_queue"}')
        == "redis-cli LLEN job_queue"
    )
    assert parse_model_command("curl localhost:3000/health") == "curl localhost:3000/health"
