# Build:  docker build -t toxic-api .
# Run:    docker run -p 8000:8000 toxic-api
# The trained artifact (src/model_artifacts/final_stack_pipeline.joblib) is NOT in git;
# create it first with `python src/train_export.py` (see README), then build.
# Use the SAME Python minor version the artifact was trained with (see model_metadata.json).
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/src/model_artifacts/final_stack_pipeline.joblib

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/app.py src/api_support.py src/web_ui.py src/toxic_pipeline.py src/
COPY src/model_artifacts/ src/model_artifacts/
# (the .joblib is git-ignored: put it in src/model_artifacts/ before building)

RUN useradd -m appuser
USER appuser
# Optional hardening, set at run time:  -e API_KEYS=key1,key2  -e RATE_LIMIT_PER_MIN=60
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
