FROM apache/airflow:3.0.6

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory '*'

USER airflow
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
# state dir is a named volume (keeps SQLite DB, logs, work files across restarts)
RUN mkdir -p /opt/airflow/state/logs /opt/airflow/state/work