# Grafana + Prometheus in ONE container, for hosts that run a single container per service
# (Render, Railway, Azure Container Apps, Cloud Run). Prometheus scrapes your deployed ECIP API
# over the internet (ECIP_URL) and Grafana shows the provisioned ECIP dashboard.
# Build from the repository root:  docker build -f deploy/monitoring/Dockerfile -t ecip-monitoring .
FROM prom/prometheus:v2.53.2 AS prom

FROM grafana/grafana:11.2.0
USER root
COPY --from=prom /bin/prometheus /usr/local/bin/prometheus
COPY monitoring/alerts.yml /etc/prometheus/alerts.yml
COPY monitoring/grafana/provisioning /etc/grafana/provisioning
COPY monitoring/grafana/dashboards /var/lib/grafana/dashboards
COPY deploy/monitoring/start.sh /usr/local/bin/ecip-monitoring-start
RUN chmod 755 /usr/local/bin/ecip-monitoring-start \
 && mkdir -p /prometheus \
 && chown -R 472:0 /prometheus /etc/prometheus /var/lib/grafana
USER 472
ENV PROMETHEUS_URL=http://127.0.0.1:9090 \
    GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/ecip.json \
    GF_AUTH_ANONYMOUS_ENABLED=true \
    GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
    GF_USERS_ALLOW_SIGN_UP=false \
    GF_ANALYTICS_REPORTING_ENABLED=false \
    GF_ANALYTICS_CHECK_FOR_UPDATES=false \
    GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES=false
ENTRYPOINT ["/usr/local/bin/ecip-monitoring-start"]
