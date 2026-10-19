#!/usr/bin/env bash
# Pull Ollama models. Works for a local Ollama or the compose container.
#   bash scripts/model_pull.sh                 # LLM_MODEL + EMBED_MODEL from .env (defaults below)
#   bash scripts/model_pull.sh llama3.1:8b     # any extra model
set -euo pipefail
[ -f .env ] && { set -a; . ./.env; set +a; }
MODELS=("${@:-${LLM_MODEL:-llama3.2:3b}}")
[ $# -eq 0 ] && MODELS+=("${EMBED_MODEL:-nomic-embed-text}")
if command -v ollama >/dev/null 2>&1; then PULL=(ollama pull)
else PULL=(docker compose exec -T ollama ollama pull); fi
for m in "${MODELS[@]}"; do echo "[models] pulling $m"; "${PULL[@]}" "$m"; done
