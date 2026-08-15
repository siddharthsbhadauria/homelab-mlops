FROM grafana/grafana:11.1.0

USER root
COPY provisioning /etc/grafana/provisioning
COPY dashboards /etc/grafana/dashboards
RUN chmod -R 777 /etc/grafana/provisioning /etc/grafana/dashboards
USER grafana
