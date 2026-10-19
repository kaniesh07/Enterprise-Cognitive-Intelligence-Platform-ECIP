# ECIP — Enterprise Cognitive Intelligence Platform v3

A private, ChatGPT-style assistant. It chats about anything **and** answers questions from your own
documents with numbered, checkable citations. It runs fully locally (Ollama + embedded vector DB),
in Docker Compose with Prometheus + Grafana monitoring, or as a single container in any cloud.

```
                     ┌──────────────── ECIP API (FastAPI, single process) ────────────────┐
 Browser / CLI ──►   │ guardrails ─► router ─┬─► chat ───────────────┐                    │
 (chat UI, SSE)      │  (input)   (no LLM,   ├─► retrieve ► grounded ─┼─► LLM (1 streamed  │──► Ollama  or any
                     │             ~1 ms)    └─► instant (small talk) │    call)           │    OpenAI-compatible API
                     │                                               └─► verify ► guardrails (output)
                     │ hybrid retrieval: Chroma vectors + BM25 → RRF (+ optional rerank)    │──► Chroma (embedded or server)
                     │ /metrics ───────────────────────────────────────────────────────────┼──► Prometheus ──► Grafana
                     └─────────────────────────────────────────────────────────────────────┘
```

## How a message is answered

| Mode (UI toggle) | Behaviour |
|---|---|
| **Auto** (default) | Small talk is answered instantly. Otherwise ECIP searches the knowledge base; if the best passages are actually relevant it answers **from the documents** with `[1]`-style citations, otherwise it answers as a **general assistant**. |
| **Chat** | General assistant, documents ignored. |
| **Documents** | Always answers from the knowledge base (says so when nothing relevant is found). |

Every answer is streamed token by token. Document answers get a fast grounding check
(`faithfulness_score`, unsupported sentences flagged) and the UI shows the sources with their snippets.

## Quick start

### A. Local, no Docker (Linux / macOS / Windows)
1. Install [Ollama](https://ollama.com/download) and start it (`ollama serve`).
2. Run:
   ```bash
   bash scripts/run_local.sh           # Windows: powershell scripts/run_local.ps1
   ```
   It creates `.venv`, installs `requirements.txt`, pulls `llama3.2:3b` + `nomic-embed-text` and starts on
   **http://localhost:8000** (chat UI at `/`, API docs at `/docs`, metrics at `/metrics`).
   The vector DB is embedded on disk in `./data` — no other services needed.

### B. Docker Compose (full stack with monitoring)
```bash
cp .env.example .env        # optional — defaults work
docker compose up -d --build
```
| Service | URL |
|---|---|
| Chat UI (nginx) | http://localhost:3000 (also served by the API at http://localhost:8000) |
| API docs | http://localhost:8000/docs |
| **Grafana** | http://localhost:3001 — `admin` / `ecip-admin` → dashboard **ECIP — Chat & RAG** |
| Prometheus | http://localhost:9090 |

The first start downloads the models (`ollama-init` service). NVIDIA GPU:
`docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`.
Bulk connector ingestion with Celery: `docker compose --profile scale up -d`.

### C. Any cloud
The root `Dockerfile` is a self-contained image (API + UI, embedded vector DB on a volume at `/app/data`,
listens on `$PORT`). Point it at an LLM:

* **OpenAI-compatible API** (OpenAI, Groq, Together, OpenRouter, Azure-compatible gateways, vLLM, LM Studio):
  `LLM_PROVIDER=openai OPENAI_BASE_URL=… OPENAI_API_KEY=… LLM_MODEL=gpt-4o-mini EMBED_PROVIDER=openai EMBED_MODEL=text-embedding-3-small`
* **Your own Ollama/GPU host**: `OLLAMA_HOST=http://gpu-box:11434`

| Platform | How |
|---|---|
| Render | `render.yaml` blueprint (disk mounted at `/app/data`) |
| Kubernetes (EKS/GKE/AKS) | `deploy/k8s/ecip.yaml` (Deployment + PVC + Service, Prometheus scrape annotations) |
| Google Cloud Run | `gcloud run deploy ecip --source . --port 8000 --set-env-vars LLM_PROVIDER=openai,…` (mount a volume or set `CHROMA_HOST` to a Chroma server for persistence) |
| AWS App Runner / ECS, Azure Container Apps, Fly.io, Railway | build & push the `Dockerfile`, expose port 8000, set the env vars above, attach persistent storage at `/app/data` |

Always set your own `API_KEYS` in production.

## Configuration (most useful settings — see `.env.example` for all)

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` (any OpenAI-compatible endpoint) |
| `LLM_MODEL` | `llama3.2:3b` | `llama3.1:8b`/`qwen2.5:7b` for better quality, `qwen2.5:1.5b` for speed |
| `EMBED_PROVIDER` / `EMBED_MODEL` | `ollama` / `nomic-embed-text` | also `openai`, `onnx` (local MiniLM folder), `hash` (tests) |
| `CHROMA_HOST` | *(empty)* | empty = embedded DB in `DATA_DIR`; set to use a Chroma server |
| `API_KEYS` | `ecip-dev-key:admin` | `key:role,…` — roles `reader` < `writer` < `admin` |
| `TOP_K_FINAL` | `4` | passages sent to the model |
| `VERIFIER_MODE` | `heuristic` | `llm` adds a second LLM fact-check pass (slower) |
| `RERANK_ENABLED` | `false` | cross-encoder rerank (needs `requirements-extras.txt`) |
| `NUM_CTX` / `MAX_TOKENS` | `4096` / `1024` | Ollama context / max answer length |

Changing the embedding model automatically uses a separate collection (no dimension clashes).

## API
```bash
KEY=ecip-dev-key
curl -N localhost:8000/query/stream -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
     -d '{"query":"How many leave days do we get?","mode":"auto","conversation_history":[]}'
curl localhost:8000/ingest -H "X-API-Key: $KEY" -F file=@handbook.pdf
curl localhost:8000/documents -H "X-API-Key: $KEY"
```
`/query/stream` emits SSE events `meta` (route + sources) → `token`… → `final` (answer, citations,
faithfulness, latency breakdown, token usage) → `done`. `/query` returns the same `final` payload as JSON.
All endpoints are also available under `/api/…`. Connectors, sync jobs, approvals, tools, user preferences
and evaluation endpoints are unchanged from v2.

Terminal chat: `make chat` (or `python scripts/query_cli.py --api-key $KEY`).

## Monitoring
`/metrics` exposes Prometheus metrics (validated with `promtool check metrics`): time-to-first-token,
end-to-end latency, per-stage latency (route / retrieval / llm / verify), LLM call latency and tokens/s per
model, token usage and projected cost, chat vs grounded traffic, error/blocked rates, guardrail events,
faithfulness and retrieval scores, ingestion, index size, LLM / vector-DB up-gauges.
Grafana is provisioned automatically (datasource + 27-panel dashboard, `monitoring/build_dashboard.py`);
Prometheus alert rules are in `monitoring/alerts.yml`.

## Latency notes
* One LLM call per question (was three: intent + reasoning + verifier); streaming starts immediately and the
  answer is generated once (the old stream endpoint generated it twice).
* Small talk is answered without the LLM; routing and verification are heuristic (≈1 ms each).
* Shared connection pools, cached Chroma client/collection, in-memory BM25, cached query embeddings,
  dense + keyword search in parallel, start-up warm-up and `keep_alive` so the model stays loaded.
* Fewer, smaller passages (300-token section-aware chunks, off-topic chunks pruned) → shorter prompts.
* Biggest remaining lever is hardware/model size: use a GPU or a smaller model (`qwen2.5:1.5b`) on CPU.

## Evaluation
```bash
make eval            # or: python scripts/run_eval.py --api http://localhost:8000 --key <key> --ingest
```
Uploads the bundled test corpus (`evaluation/datasets/corpus`: handbook, PDF, DOCX, CSV, 3-page website),
asks the 35 questions in `evaluation/datasets/eval_set.json` through the real streaming API and reports routing
accuracy, retrieval Hit@k / MRR / nDCG, answer accuracy (required facts present), citation accuracy, grounding
score and latency per question type, plus a dense vs BM25 vs hybrid retrieval ablation. Results are written to
`evaluation/results/latest.md` / `.json`. Point `--api` at a cloud deployment to evaluate it the same way.

## Development
```bash
make venv && make test      # 262 tests, no external services needed
make lint
```
Project layout: `api/` (FastAPI + SSE), `agents/` (pipeline, LLM client, router, circuit breaker),
`retrieval/` (embeddings, hybrid retriever), `ingestion/` (parsers, chunker, indexer), `guardrails/`,
`monitoring/` (metrics, Prometheus, Grafana), `frontend/` (single-file chat UI), `connectors/`, `pipeline/`
(bulk sync), `tools/`, `memory/`, `evaluation/`, `deploy/`.

See `CHANGELOG.md` for the bugs fixed in v3 and `docs/verification/REPORT.md` for the sandbox test run (benchmarks, screenshots, monitoring checks).
