from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from wargames_env.models import StepResult, WarGamesAction, WarGamesObservation
from wargames_env.server.env import WarGamesEnv


@asynccontextmanager
async def lifespan(app: FastAPI):
    env = WarGamesEnv()
    app.state.env = env
    try:
        yield
    finally:
        env.close()


app = FastAPI(
    title="WarGames OpenEnv",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/reset", response_model=WarGamesObservation)
async def reset(task_name: str | None = None) -> WarGamesObservation:
    try:
        env: WarGamesEnv = app.state.env
        return env.reset(task_name=task_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/step", response_model=StepResult)
async def step(action: WarGamesAction) -> StepResult:
    try:
        env: WarGamesEnv = app.state.env
        return env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state")
async def state() -> dict[str, object]:
    try:
        env: WarGamesEnv = app.state.env
        return env.state()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
