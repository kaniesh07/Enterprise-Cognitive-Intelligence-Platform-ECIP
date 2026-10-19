.PHONY: help venv install install-extras dev run run-local test lint up up-build up-gpu up-scale down down-volumes \
        logs logs-api health pull-model chat query ingest-dir dashboard docker-build clean

PY      ?= python3
VENV    ?= .venv
VPY      = $(VENV)/bin/python
API     ?= http://localhost:8000
API_KEY ?= ecip-dev-key

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[32m%-16s\033[0m %s\n", $$1, $$2}'

# ── Local (no Docker) ──────────────────────────────────────────────────────
venv: ## Create .venv and install core + dev dependencies
	$(PY) -m venv $(VENV)
	$(VPY) -m pip install -U pip
	$(VPY) -m pip install -r requirements-dev.txt

install: ## pip install core requirements into the current environment
	pip install -r requirements.txt

install-extras: ## pip install optional features (OCR, connectors, reranker, Celery, RAGAS)
	pip install -r requirements-extras.txt

env: ## Create .env from .env.example (if missing)
	@[ -f .env ] && echo ".env already exists" || (cp .env.example .env && echo "Created .env")

run-local: env ## Run API + UI locally (needs Ollama running: https://ollama.com/download)
	bash scripts/run_local.sh

run: run-local

dev: env ## Run with auto-reload
	$(VPY) -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run the test suite
	$(VPY) -m pytest tests

lint: ## Ruff lint
	$(VPY) -m ruff check .

# ── Docker Compose ─────────────────────────────────────────────────────────
up: env ## Start the full stack (Ollama, Chroma, API, UI, Prometheus, Grafana)
	docker compose up -d --build
	@echo "Chat UI http://localhost:3000 | API http://localhost:8000/docs | Grafana http://localhost:3001"

up-gpu: env ## Same as `up` with NVIDIA GPU for Ollama
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

up-scale: env ## Also start Redis + Celery worker + Flower
	docker compose --profile scale up -d --build

down: ## Stop the stack
	docker compose --profile scale down

down-volumes: ## Stop and DELETE all data volumes
	docker compose --profile scale down -v

logs: ## Tail all logs
	docker compose logs -f

logs-api: ## Tail API logs
	docker compose logs -f api

pull-model: ## Pull another Ollama model: make pull-model MODEL=llama3.1:8b
	docker compose exec ollama ollama pull $(MODEL)

docker-build: ## Build the API image (for cloud deploys)
	docker build -t ecip-api:latest .

# ── Usage ──────────────────────────────────────────────────────────────────
health: ## API health
	@curl -sf $(API)/health | $(PY) -m json.tool || echo "API unreachable at $(API)"

chat: ## Interactive terminal chat
	ECIP_API=$(API) ECIP_API_KEY=$(API_KEY) $(PY) scripts/query_cli.py

query: ## One question: make query Q="What is our leave policy?"
	ECIP_API=$(API) ECIP_API_KEY=$(API_KEY) $(PY) scripts/query_cli.py --query "$(Q)"

ingest-dir: ## Index a folder through the API: make ingest-dir DIR=./docs
	@for f in $$(find $(DIR) -type f \( -name '*.pdf' -o -name '*.docx' -o -name '*.txt' -o -name '*.md' -o -name '*.csv' -o -name '*.html' \)); do \
	  echo "→ $$f"; curl -sf -H "X-API-Key: $(API_KEY)" -F "file=@$$f" $(API)/ingest || echo "  failed"; echo; done

dashboard: ## Regenerate the Grafana dashboard JSON
	$(PY) monitoring/build_dashboard.py

gen-key: ## Generate a random API key
	@$(PY) -c "import secrets; print(secrets.token_urlsafe(32))"

clean: ## Remove caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} +; rm -rf .pytest_cache .ruff_cache

# ── Connectors (go through the running API) ────────────────────────────────
connector-types: ## List supported connector types
	@curl -sf -H "X-API-Key: $(API_KEY)" $(API)/connectors | $(PY) -m json.tool

connector-register: ## Register: make connector-register NAME=x TYPE=s3 CONFIG='{"bucket_name":"b"}'
	@curl -sf -X POST -H "X-API-Key: $(API_KEY)" -H "Content-Type: application/json" $(API)/connectors \
	  -d '{"name":"$(NAME)","connector_type":"$(TYPE)","config":$(CONFIG)}' | $(PY) -m json.tool

connector-register-local: ## Register a folder: make connector-register-local NAME=docs DIR=/path
	@$(MAKE) -s connector-register NAME=$(NAME) TYPE=local CONFIG='{"root_path":"$(DIR)","recursive":true}'

connector-register-web: ## Register a website: make connector-register-web NAME=site URL=http://localhost:8080 DEPTH=3
	@$(MAKE) -s connector-register NAME=$(NAME) TYPE=web CONFIG='{"urls":["$(URL)"],"max_depth":$(or $(DEPTH),3),"max_pages":500,"same_domain":true}'

connector-sync: ## Sync (incremental): make connector-sync NAME=x   (FORCE=true to re-index all)
	@curl -sf -X POST -H "X-API-Key: $(API_KEY)" -H "Content-Type: application/json" $(API)/connectors/$(NAME)/sync \
	  -d '{"force":$(or $(FORCE),false)}' | $(PY) -m json.tool

sync-status: ## Show sync jobs (or one: make sync-status JOB=<id>)
	@curl -sf -H "X-API-Key: $(API_KEY)" $(API)/sync/jobs$(if $(JOB),/$(JOB),) | $(PY) -m json.tool

sync-manifest: ## Indexed / failed / skipped counts
	@curl -sf -H "X-API-Key: $(API_KEY)" $(API)/sync/manifest/stats | $(PY) -m json.tool

eval: ## Evaluate the running API (uploads the eval corpus) + retrieval ablation → evaluation/results/latest.md
	$(PY) scripts/run_eval.py --api $(API) --key $(API_KEY) --ingest

ops-eval: ## Ingestion, connector, load and failure tests against the running API
	$(PY) -m evaluation.ops_eval --api $(API) --key $(API_KEY)

guardrail-eval: ## Safety-filter block / false-positive rates (no server needed)
	$(PY) -m evaluation.guardrail_eval
