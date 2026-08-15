FROM grafana/grafana:11.1.0

USER root
COPY provisioning /etc/grafana/provisioning
COPY dashboards /var/lib/grafana/dashboards
RUN chown -R grafana:root /etc/grafana/provisioning /var/lib/grafana/dashboards
USER grafana
