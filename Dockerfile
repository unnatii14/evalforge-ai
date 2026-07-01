# EvalForge application image.
#
# Default entrypoint runs the Streamlit dashboard on port 7860 (the Hugging Face
# Spaces default), seeding demo data first so the app is never empty. The image
# defaults to the Groq evaluation backend for cloud use; docker-compose overrides
# this to Ollama and runs the API/dashboard as separate services locally.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    EVAL_BACKEND=groq \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching.
COPY pyproject.toml ./
COPY app ./app
RUN pip install --upgrade pip && pip install .

# Application assets: sample datasets, startup + seed scripts.
COPY data ./data
COPY scripts ./scripts

# Run as an unprivileged user (uid 1000, as expected by Hugging Face Spaces).
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data /app/.cache/huggingface \
    && chmod +x scripts/start.sh \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860 8000

CMD ["bash", "scripts/start.sh"]
