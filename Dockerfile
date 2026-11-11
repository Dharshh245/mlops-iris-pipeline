# Q2 (a) - Container image for the Iris model API
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.joblib

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
# The model is produced by the DVC pipeline (`dvc repro`) before `docker build`
COPY models/model.joblib models/model.joblib

# Set at build time by CI to the Git commit SHA
ARG MODEL_VERSION=dev
ENV MODEL_VERSION=${MODEL_VERSION}

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
