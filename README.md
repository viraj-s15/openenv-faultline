# WarGames

WarGames is an OpenEnv environment for teaching agents to interact with a live distributed system through bash commands.

Phase 0 ports the Round 1 service mesh into a root-deployable project:

- Gateway on port `3000`
- Auth on port `3001`
- Redis on port `6379`
- OpenEnv API on port `8000`

The Python package lives under `src/wargames_env`, and the mesh services live under `mesh`.
