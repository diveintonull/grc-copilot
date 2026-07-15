# GRC Copilot — Enterprise Compliance Knowledge Base Agent

GRC Copilot is an evidence-oriented compliance Agent prototype with three work modes: regulation Q&A, clause comparison, and control gap analysis. It exposes observable Server-Sent Events for answers, references, recommendations, execution Trace, terminal status, and cancellation.

## Reproducible Docker demo

The Compose stack contains exactly two services:

- `app`: FastAPI, the observable browser UI, and a deterministic demo runner.
- `qdrant`: the vector database used by the retrieval architecture.

Postgres is intentionally absent. This version does not persist accounts, conversations, jobs, or audit records in relational tables, so adding an unused database would create operational complexity without owning any data.

### Prerequisites

- Docker Desktop or Docker Engine with Docker Compose v2.
- Enough disk space for the locked Python container runtime image.

### Start

Copy the environment template without adding secrets:

```powershell
Copy-Item .env.example .env
```

Build and start the two services:

```powershell
docker compose up --build --wait
```

Open <http://127.0.0.1:8000>. Qdrant's local dashboard is available at <http://127.0.0.1:6333/dashboard>.

Check service state:

```powershell
docker compose ps
curl.exe http://127.0.0.1:8000/ready
curl.exe http://127.0.0.1:6333/readyz
```

Both containers should report `healthy`. Compose waits for Qdrant's `/readyz` probe before it starts the app. The app then checks Qdrant through `http://qdrant:6333` before `/ready` succeeds or `/chat` accepts work.

### Regulation Q&A smoke test

The default `APP_RUN_MODE=demo` is deterministic and does not require an API key:

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/chat `
  -H "Content-Type: application/json" `
  -d '{"request_id":"compose-qa","mode":"regulation_qa","query":"管理员身份鉴别有哪些要求？"}'
```

The stream should contain `status`, one or more `text` events, a `reference` for `GBT-22239@2019#8.1.4.1`, observable `trace` events, and a terminal `done` event.

### Cancellation smoke test

Start a deliberately slow request in one terminal:

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/chat `
  -H "Content-Type: application/json" `
  -d '{"request_id":"compose-stop","mode":"regulation_qa","query":"慢速法规问答，用于停止验收"}'
```

Then stop the same request from a second terminal:

```powershell
curl.exe -X POST http://127.0.0.1:8000/tasks/compose-stop/stop
```

The original stream should end with `error`, `status=cancelled`, and `code=request_cancelled`. It must not report a successful `done` event.

### Persistence and model cache

Compose uses two named volumes:

- `qdrant_storage` is mounted at `/qdrant/storage`, so vector data survives container replacement.
- `model_cache` is mounted at `/home/app/.cache`, so Linux-container model downloads can survive app rebuilds.

Named volumes are used instead of Windows bind mounts because Qdrant requires POSIX-compatible block storage and warns that Docker/WSL bind mounts on Windows can cause filesystem problems. A Windows host's existing ModelScope cache is not automatically reusable inside Linux containers; the container cache has different paths and binary compatibility expectations.

Normal shutdown preserves both volumes:

```powershell
docker compose down
```

Deleting volumes is destructive and removes the Qdrant index and cached models:

```powershell
docker compose down --volumes
```

## What the Docker demo proves

The default Compose mode proves that a clean checkout can reproduce:

- the app and Qdrant startup order;
- liveness and dependency readiness;
- the browser UI and stable SSE contract;
- evidence cards and observable Trace;
- task cancellation and terminal cleanup;
- persistent Qdrant and model-cache locations.

The demo runner returns explicit fixtures. It does **not** claim that a model generated the answer or that Qdrant retrieved the fixture. The repository intentionally excludes licensed/private corpus files, built indexes, and local model caches; the current codebase also has no production LLM/Graph composition root. A real deployment must provide governed corpus data, build the index, configure an OpenAI-compatible endpoint, and inject a real Agent runner instead of relabeling the fixture.

The Dockerfile installs only the locked `container` dependency group (FastAPI, uvicorn, and the Qdrant client). Local parsing, evaluation, MinerU, and model dependencies remain available through the normal project environment but are not copied into the deterministic demo image.

## Local development

Install the locked environment and run the full test suite:

```powershell
uv sync --locked
uv run pytest -p no:cacheprovider -q
```

Run the deterministic deployment app against a local Qdrant instance:

```powershell
$env:QDRANT_URL = "http://127.0.0.1:6333"
uv run uvicorn api.deployment:app --host 127.0.0.1 --port 8000
```

The lower-level `api.main:app` keeps an intentionally unconfigured runner. This makes missing runtime composition fail explicitly instead of silently substituting demo answers.

## Repository layout

- `agent/`: LangGraph state, nodes, Skills, and local/MCP tool adapters.
- `api/`: stable SSE events, task cancellation, FastAPI, and deployment composition.
- `ingest/`: versioned parsing, parent-child chunking, embedding, and indexing.
- `rag/`: dense/sparse retrieval, reranking, grounded generation, and citation checks.
- `mcp_server/`: MCP adapters over the deterministic GRC tools.
- `evals/`: datasets, metrics, and ablation runners.
- `skills/`: progressively disclosed GRC workflow instructions.
- `web/`: observable three-mode browser UI.
