# ECIP API + chat UI — one image for local Docker Compose and any cloud container platform
# (Cloud Run, App Runner/ECS, Azure Container Apps, Render, Railway, Fly.io, Kubernetes).
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ANONYMIZED_TELEMETRY=False \
    LOG_FORMAT=json \
    DATA_DIR=/app/data \
    PORT=8000

WORKDIR /app

# EXTRAS=true adds OCR (tesseract), connectors, Celery, reranker, evaluation (much bigger image)
ARG EXTRAS=false
RUN if [ "$EXTRAS" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
      && rm -rf /var/lib/apt/lists/*; fi

COPY requirements.txt requirements-extras.txt ./
RUN pip install -r requirements.txt \
 && if [ "$EXTRAS" = "true" ]; then pip install -r requirements-extras.txt; fi

COPY . .

RUN useradd --create-home --uid 10001 ecip \
 && mkdir -p /app/data/raw /app/data/processed /app/data/chroma /app/evaluation/results \
 && chown -R ecip:ecip /app
USER ecip

EXPOSE 8000
VOLUME ["/app/data"]

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)" || exit 1

# Single process on purpose: the BM25 index, metrics and model warm-up live in-process.
# Scale out with more replicas instead of more workers.
CMD ["python", "-m", "api.main"]
