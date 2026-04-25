from evals.run_red_blue_benchmark import (
    STEP_FIELDS,
    classify_services_affected,
    classify_services_restored,
)


def test_step_fields_include_eval_only_service_impact_columns():
    assert "services_affected" in STEP_FIELDS
    assert "services_restored" in STEP_FIELDS


def test_classify_services_affected_from_red_command():
    services = classify_services_affected(
        "redis-cli set LOCK:job_processor attacker EX 3600 && "
        'echo \'{"db_pool_size":1,"db_write_delay_ms":30000}\' > '
        "/home/user/app/mesh/worker/config.json"
    )

    assert services == "redis,worker"


def test_classify_services_restored_from_blue_actions():
    services = classify_services_restored(
        [
            {
                "kind": "llm_command",
                "target": "redis-cli del LOCK:job_processor && "
                'echo \'{"db_pool_size":10,"db_write_delay_ms":0}\' > '
                "/home/user/app/mesh/worker/config.json && kill -HUP 21",
                "status": "applied",
                "detail": "restored worker config",
            }
        ]
    )

    assert services == "redis,worker"
