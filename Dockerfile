FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    jq \
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
COPY --chown=user:user src/ ./src/
COPY --chown=user:user server/ ./server/
RUN pip install --no-cache-dir -e .

COPY --chown=user:user mesh/ ./mesh/
RUN cd mesh/gateway && bun install
RUN cd mesh/auth && bun install
RUN cd mesh/worker && bun install

COPY --chown=user:user start.sh ./
RUN chmod +x ./start.sh

EXPOSE 8000
CMD ["./start.sh"]
