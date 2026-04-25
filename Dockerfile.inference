# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    dnsutils \
    iproute2 \
    iputils-ping \
    jq \
    lsof \
    net-tools \
    procps \
    redis-server \
    sqlite3 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /home/user/app
RUN rm -rf /mesh && ln -s /home/user/app/mesh /mesh && chown -h user:user /mesh && chown -R user:user /home/user
USER user

ENV HOME=/home/user
ENV PATH="/home/user/.bun/bin:/home/user/.local/bin:${PATH}"

RUN curl -fsSL https://bun.sh/install | bash

COPY --chown=user:user pyproject.toml README.md openenv.yaml ./
RUN --mount=type=cache,target=/home/user/.cache/pip,uid=1000,gid=1000 \
    python - <<'PY'
import subprocess
import tomllib
from pathlib import Path

dependencies = tomllib.loads(Path("pyproject.toml").read_text())["project"]["dependencies"]
subprocess.check_call(["python", "-m", "pip", "install", "--user", *dependencies])
PY

RUN mkdir -p mesh/gateway mesh/auth mesh/worker
COPY --chown=user:user mesh/gateway/package.json mesh/gateway/bun.lock ./mesh/gateway/
COPY --chown=user:user mesh/auth/package.json mesh/auth/bun.lock ./mesh/auth/
COPY --chown=user:user mesh/worker/package.json mesh/worker/bun.lock ./mesh/worker/
RUN --mount=type=cache,target=/home/user/.bun/install/cache,uid=1000,gid=1000 \
    cd mesh/gateway && bun install --frozen-lockfile
RUN --mount=type=cache,target=/home/user/.bun/install/cache,uid=1000,gid=1000 \
    cd mesh/auth && bun install --frozen-lockfile
RUN --mount=type=cache,target=/home/user/.bun/install/cache,uid=1000,gid=1000 \
    cd mesh/worker && bun install --frozen-lockfile

COPY --chown=user:user src/ ./src/
COPY --chown=user:user server/ ./server/
COPY --chown=user:user inference.py ./
RUN --mount=type=cache,target=/home/user/.cache/pip,uid=1000,gid=1000 \
    pip install --user --no-deps -e .

COPY --chown=user:user mesh/ ./mesh/

COPY --chown=user:user start.sh ./
RUN chmod +x ./start.sh

EXPOSE 8000
CMD ["./start.sh"]
