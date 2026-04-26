import time

import httpx

from wargames_env.models import StepResult, WarGamesAction, WarGamesObservation, WarGamesState


_RETRY_STATUS = {500, 502, 503, 504}
_RETRY_DELAYS_S = (1.0, 3.0, 8.0, 20.0)  # ~32s total wall clock


class EnvUnavailableError(RuntimeError):
    """Raised when the env Space is permanently unreachable for a single call.

    Trainer catches this to emit a synthetic dead-step instead of aborting the
    whole run. Distinguished from `httpx.HTTPStatusError` so callers don't
    accidentally retry on 4xx (caller bug, not transient).
    """


def _is_retriable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRY_STATUS
    return False


class WarGamesTrainingClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=45.0)

    def close(self) -> None:
        self._client.close()

    def _request_with_retry(self, method: str, path: str, **kw):
        last_exc: BaseException | None = None
        for attempt, delay in enumerate((0.0, *_RETRY_DELAYS_S)):
            if delay:
                time.sleep(delay)
            try:
                response = self._client.request(method, path, **kw)
                response.raise_for_status()
                return response
            except (httpx.HTTPStatusError, httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
                if not _is_retriable(exc):
                    raise
        raise EnvUnavailableError(
            f"{method} {path} failed after {len(_RETRY_DELAYS_S) + 1} attempts: {last_exc}"
        ) from last_exc

    def reset(self, task_name: str) -> WarGamesObservation:
        response = self._request_with_retry("POST", "/reset", params={"task_name": task_name})
        return WarGamesObservation.model_validate(response.json())

    def step(self, command: str) -> StepResult:
        payload = WarGamesAction(command=command).model_dump()
        response = self._request_with_retry("POST", "/step", json=payload)
        return StepResult.model_validate(response.json())

    def state(self) -> WarGamesState:
        response = self._request_with_retry("GET", "/state")
        return WarGamesState.model_validate(response.json())