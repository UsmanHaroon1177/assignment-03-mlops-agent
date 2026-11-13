FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first so this layer is cached between code changes.
COPY serving/requirements.txt serving/requirements.txt
RUN pip install --no-cache-dir -r serving/requirements.txt

COPY api/ api/
COPY serving/model/ serving/model/

# Run as a non-root user (numeric UID so Kubernetes runAsNonRoot can verify it).
RUN useradd --system --uid 10001 appuser
USER 10001

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]