# Changelog


### Deploy fix (18 Oct 2026)
* Cloud deploys left on the default `LLM_PROVIDER=ollama` / `OLLAMA_HOST=localhost` failed with
  "All connection attempts failed". Start-up now logs `model_backend_unreachable` with the exact settings to
  change, and the chat shows the same explanation instead of the raw connection error.
* `API_KEYS` now tolerates quotes, spaces, upper-case roles and a bare key (treated as admin); start-up logs
  `api_keys_loaded` or `api_keys_none_valid`. The chat page opens Settings automatically on a 401 and strips a
  pasted `:admin` suffix from the key.
* Groq retired `llama-3.1-8b-instant` (16 Aug 2026): free config now uses `openai/gpt-oss-20b`. New
  `REASONING_EFFORT` setting (sent as `reasoning_effort` for reasoning models); a retired/unknown model now gives a
  clear "set LLM_MODEL" message instead of a raw HTTP 404.
* Added `render.free.yaml` (free plan, Groq LLM, hash embeddings).

## v3.0.0 — ChatGPT-style chat, single-pass streaming, monitoring, local + cloud packaging

### Bugs fixed
| Area | Bug | Fix |
|---|---|---|
| Chat | Anything not matching a short greeting regex went to document search; with no documents every question got "upload a file". | Router: small talk → instant; empty index or irrelevant documents → general chat; relevant documents → grounded answer. UI toggle Auto / Chat / Documents. |
| Chat | `"hi"`, `"ok"` were rejected with HTTP 400 (3-character minimum in the input guardrail). | Minimum length 1. |
| Latency | 3 sequential LLM calls per question (intent JSON, reasoning with `<thinking>`, verifier). | 1 streamed call; heuristic intent and grounding check (LLM verifier optional). |
| Latency | `/query/stream` ran the whole pipeline, then generated the answer **a second time**. | Retrieval once, then a single streamed generation. |
| Latency | New Chroma HTTP client + BM25 pickle load **and rebuild** on every query; one HTTP call per embedding; no connection reuse. | Shared clients and collection, in-memory BM25 (auto-reload on change), batched `/api/embed`, pooled keep-alive connections, cached query embeddings, parallel dense + BM25 search. |
| Latency | Start-up waited up to 2 min polling a Chroma `/api/v2/heartbeat` that 0.5.23 does not have; `/health` retried sequentially (up to ~6 s). | Non-blocking warm-up (loads model, embeddings, index); parallel 2 s health probes. |
| Latency | tiktoken re-attempted its vocabulary download on every token count when offline. | Encoder loaded once with fallback. |
| Retrieval | rank_bm25's default IDF is ≤ 0 for terms in more than half the chunks — small knowledge bases got **no** keyword hits. | Lucene-style positive IDF. |
| Retrieval | Text without sentence punctuation became one oversized chunk (embedding models truncate it). | Long-sentence splitting, Markdown-heading-aware chunks, 300-token default. |
| Retrieval | CSV rows were indexed twice (table + text). | Indexed once as `header: value` rows. |
| Ingestion | Uploaded documents were titled with the temp-file name (`tmpa8x9…`), so citations were meaningless. | Original filename is the title (API and connector sync). |
| Ingestion | Re-uploading a document kept stale chunks; deleting left orphan BM25 entries that were still retrieved. | Re-ingest replaces, delete removes from both indexes; atomic BM25 writes. |
| Ingestion | Parallel connector sync threads each loaded and overwrote the BM25 pickle (lost updates); progress counters raced. | Single write lock; atomic progress increments. |
| Ingestion | Docling ran first for every file (slow, downloads models). | Opt-in `USE_DOCLING=true`. |
| Web connector | Links were collected *after* `<nav>/<header>/<footer>` were stripped, so crawls stopped at the first page of most sites; pages with the same `<title>` overwrote each other. | Links collected first; page path added to the title. |
| Connectors | A dot in a document title ("v2.0 notes") produced a bogus file extension; synced docs got mangled titles. | Real extensions only; original titles kept. |
| API | `DELETE /documents/{title}` failed for titles containing `/`. | Path-style parameter. |
| Concurrency | `top_k` was passed by mutating `os.environ` (shared across concurrent requests). | Passed as an argument. |
| Metrics | Hand-written exporter emitted duplicate `# HELP/# TYPE` lines for labelled series → Prometheus rejected the scrape, Grafana stayed empty. | `prometheus_client`, labelled families, validated with `promtool check metrics`. |
| Metrics | `token_usage` per request was the process-wide cumulative total. | Per-request accounting. |
| Grafana | Datasource had no UID; dashboards and provider config lived in the same folder; old metric names. | Provisioned datasource `uid: prometheus`, new 27-panel dashboard generated from code, alert rules. |
| Auth | Global API-key middleware rejected CORS pre-flight `OPTIONS` and static files. | Per-endpoint dependencies only; constant-time key comparison. |
| Guardrails | False positives: "what is a system prompt?", "ignore case in a regex filter", any mention of "suicide prevention policy" (operator-precedence bug), "attack them" in normal text. | Narrowed patterns; supportive response for real self-harm requests. |
| Guardrails | Low faithfulness replaced the entire answer; hedge/citation checks also ran on general chat answers. | Warn instead of hide; grounding checks only for document answers. |
| UI | Not streaming; no Markdown; XSS via document titles in `innerHTML`/`onclick`; API URL hard-coded to `localhost:8000`. | New ChatGPT-style UI: streaming, stop/regenerate/copy, chat history, Markdown + code blocks, source chips, drag-and-drop upload, escaped output, same-origin `/api`. |
| CLI | `scripts/query_cli.py` never sent `X-API-Key` (every call 401). | Sends key (`--api-key` / `ECIP_API_KEY`), supports `--mode`. |
| Docker | `bootstrap.sh` used `wget` in images that do not have it; models had to be pulled by hand; API source bind-mounted over the image; Celery worker listened on queue `ecip` that tasks were never sent to. | `ollama-init` pulls models automatically; images are self-contained; Celery default queue set to `ecip`. |
| Packaging | `pyproject.toml` referenced a non-existent build backend; one huge requirements file (LangChain, LangGraph, Docling, RAGAS…) for the API. | Core `requirements.txt` (no LangChain/LangGraph), optional `requirements-extras.txt`, dev requirements. |

### Added
* OpenAI-compatible LLM + embedding providers (cloud), local ONNX embeddings, embedded Chroma mode.
* `/query/stream` SSE protocol with `meta` / `token` / `final` events, `/ready`, `/config/public`, `/api/*` alias.
* `scripts/run_local.sh` / `.ps1`, root `Dockerfile` (non-root, healthcheck, `$PORT`), GPU compose override,
  Render blueprint, Kubernetes manifest, Prometheus alert rules.
* Makefile connector targets that work through the running API (`connector-register-web`, `connector-sync`, …);
  `host.docker.internal` reachable from the API container for sites/services on your machine.
* `tests/test_chat_pipeline.py` (16 end-to-end tests with an in-memory store and fake LLM).
