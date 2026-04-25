import httpx

from wargames_env.models import StepResult, WarGamesAction, WarGamesObservation, WarGamesState


class WarGamesTrainingClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=45.0)

    def close(self) -> None:
        self._client.close()

    def reset(self, task_name: str) -> WarGamesObservation:
        response = self._client.post("/reset", params={"task_name": task_name})
        response.raise_for_status()
        return WarGamesObservation.model_validate(response.json())

    def step(self, command: str) -> StepResult:
        payload = WarGamesAction(command=command).model_dump()
        response = self._client.post("/step", json=payload)
        response.raise_for_status()
        return StepResult.model_validate(response.json())

    def state(self) -> WarGamesState:
        response = self._client.get("/state")
        response.raise_for_status()
        return WarGamesState.model_validate(response.json())
