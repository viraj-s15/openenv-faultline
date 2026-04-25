import subprocess

import httpx

from wargames_env.server.metrics_poller import MetricsPoller


def test_failed_gateway_poll_degrades_gateway_metrics(monkeypatch):
    class FailingClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            raise httpx.ConnectError("gateway down")

    monkeypatch.setattr(httpx, "Client", FailingClient)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args, returncode=0, stdout="0\n", stderr=""
        ),
    )
    poller = MetricsPoller()

    poller.poll_once()
    metrics = poller.get_current_metrics()

    assert metrics.gateway_success_rate == 0.0
    assert metrics.gateway_p99_latency_ms >= 5000.0
