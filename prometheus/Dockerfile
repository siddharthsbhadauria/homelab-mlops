FROM prom/prometheus:v2.53.0

USER root
COPY prometheus.yml /etc/prometheus/prometheus.yml
RUN chmod -R 777 /etc/prometheus
USER nobody
