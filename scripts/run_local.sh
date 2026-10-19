#!/usr/bin/env bash
# Run ECIP on this machine without Docker:
#   * embedded Chroma vector DB in ./data (no server)
#   * Ollama for the LLM + embeddings (https://ollama.com/download)
# Usage: bash scripts/run_local.sh            (reads .env if present)
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || cp .env.example .env
set -a; . ./.env; set +a

PY="${PYTHON:-python3}"
if [ ! -d .venv ]; then
  echo "[ecip] creating virtualenv .venv"
  "$PY" -m venv .venv
  .venv/bin/pip install -q -U pip
  .venv/bin/pip install -q -r requirements.txt
fi

if [ "${LLM_PROVIDER:-ollama}" = "ollama" ]; then
  HOST="${OLLAMA_HOST:-http://localhost:11434}"
  if ! curl -sf "$HOST/api/tags" >/dev/null; then
    echo "[ecip] Ollama is not reachable at $HOST."
    echo "       Install it from https://ollama.com/download and run: ollama serve"
    exit 1
  fi
  for m in "${LLM_MODEL:-llama3.2:3b}" $( [ "${EMBED_PROVIDER:-ollama}" = "ollama" ] && echo "${EMBED_MODEL:-nomic-embed-text}" ); do
    if ! curl -sf "$HOST/api/tags" | grep -q "\"name\":\"$m"; then
      echo "[ecip] pulling $m (first run only)…"
      OLLAMA_HOST="$HOST" ollama pull "$m"
    fi
  done
fi

export CHROMA_HOST="${CHROMA_HOST:-}"
echo "[ecip] starting on http://localhost:${PORT:-8000}  (API docs: /docs, metrics: /metrics)"
exec .venv/bin/python -m api.main
