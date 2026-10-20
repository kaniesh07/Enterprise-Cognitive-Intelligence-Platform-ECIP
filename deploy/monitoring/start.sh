#!/bin/sh
# Start Prometheus (internal, 127.0.0.1:9090) scraping ECIP_URL, then Grafana on $PORT.
set -e
if [ -z "$ECIP_URL" ]; then
  echo "ERROR: set ECIP_URL to your ECIP API address, e.g. https://ecip-free.onrender.com" >&2
  exit 1
fi
url=${ECIP_URL%/}
case "$url" in
  http://*|https://*) ;;
  *) url="https://$url" ;;          # bare host name -> https
esac
scheme=${url%%://*}
rest=${url#*://}
target=${rest%%/*}
prefix=""
[ "$rest" != "$target" ] && prefix="/${rest#*/}"
case "$target" in *:*) ;; *) [ "$scheme" = https ] && target="$target:443" || target="$target:80" ;; esac

PROM_CFG=${PROM_CFG:-/etc/prometheus/prometheus.yml}
cat > "$PROM_CFG" <<CFG
global:
  scrape_interval: ${SCRAPE_INTERVAL:-30s}
  scrape_timeout: ${SCRAPE_TIMEOUT:-${SCRAPE_INTERVAL:-30s}}
  evaluation_interval: 30s
  external_labels:
    project: ecip
rule_files:
  - /etc/prometheus/alerts.yml
scrape_configs:
  - job_name: ecip-api
    scheme: $scheme
    metrics_path: ${prefix}/metrics
    static_configs:
      - targets: ["$target"]
  - job_name: prometheus
    static_configs:
      - targets: ["127.0.0.1:9090"]
CFG
echo "Prometheus will scrape $scheme://$target${prefix}/metrics every ${SCRAPE_INTERVAL:-30s}"

prometheus --config.file="$PROM_CFG" \
  --storage.tsdb.path="${PROM_DATA:-/prometheus}" \
  --storage.tsdb.retention.time="${RETENTION:-3d}" \
  --web.listen-address=127.0.0.1:9090 &
PROM_PID=$!
sleep 3
if ! kill -0 "$PROM_PID" 2>/dev/null; then
  echo "ERROR: Prometheus failed to start (see the message above)" >&2
  exit 1
fi

export PROMETHEUS_URL="${PROMETHEUS_URL:-http://127.0.0.1:9090}"
[ -n "$PORT" ] && export GF_SERVER_HTTP_PORT="$PORT"
[ -n "$GRAFANA_PASSWORD" ] && export GF_SECURITY_ADMIN_PASSWORD="$GRAFANA_PASSWORD"
exec "${GRAFANA_RUN:-/run.sh}"
