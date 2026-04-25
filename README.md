# WarGames

WarGames is an OpenEnv environment for teaching agents to interact with a live distributed system through bash commands.

Phase 0 ports the Round 1 service mesh into a root-deployable project. Phase 1 gives the Red agent raw bash access through the `command` action field.

- Gateway on port `3000`
- Auth on port `3001`
- Redis on port `6379`
- OpenEnv API on port `8000`

The Python package lives under `src/wargames_env`, and the mesh services live under `mesh`.

## Local Run

```bash
APP_ROOT="$PWD" MESH_ROOT="$PWD/mesh" ./start.sh
```

Then call the environment:

```bash
curl -X POST http://localhost:8000/reset
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"command": "curl -sf localhost:3000/health"}'
```

In Docker, `/mesh` points at the app-local mesh directory. On local machines, `start.sh` attempts to create the same `/mesh` link when permissions allow it. If that link is unavailable, use the exported `MESH_ROOT` path inside Red commands, for example `cat "$MESH_ROOT/gateway/config.json"`.

## Red Action Schema

The Red agent sends a single raw bash command:

```json
{"command": "redis-cli KEYS '*'"}
```

`/step` executes the command with `subprocess.run(command, shell=True)`. The response includes merged stdout/stderr in `observation.command_output`, current metrics, process status, and command metadata in `info`:

- `exit_code`
- `timed_out`
- `command`
- `duration_ms`

## Phase 1 Example Commands

Recon:

```bash
cat /mesh/gateway/config.json
redis-cli KEYS '*'
tail -20 /tmp/worker.log
curl localhost:3000/health
```

Attack:

```bash
redis-cli LPUSH job_queue '{broken'
echo '{"delay_ms": 1500}' > /mesh/auth/config.json
kill -9 $(pgrep worker)
```

Stealth:

```bash
truncate -s 0 /tmp/worker.log
```

Phase 1 intentionally allows destructive commands inside the isolated environment. Later phases add Blue defense and reward logic.
